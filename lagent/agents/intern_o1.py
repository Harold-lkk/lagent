import json
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Union

from lagent.agents.agent import Agent, AsyncAgent
from lagent.agents.aggregator import DefaultAggregator
from lagent.agents.streaming import AsyncStreamingAgent, StreamingAgent
from lagent.hooks import MessageLogger
from lagent.llms import AsyncLMDeployClient, BaseLLM, LMDeployClient
from lagent.schema import AgentMessage

INTERNLM2_META = [
    dict(
        role='system',
        begin=dict(
            with_name='<|im_start|>system name={name}\n',
            without_name='<|im_start|>system\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            },
        ),
        end='<|im_end|>\n',
    ),
    dict(
        role='user',
        begin=dict(
            with_name='<|im_start|>user name={name}\n',
            without_name='<|im_start|>user\n',
        ),
        end='<|im_end|>\n',
    ),
    dict(
        role='assistant',
        begin=dict(
            with_name='<|im_start|>assistant name={name}\n',
            without_name='<|im_start|>assistant\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            },
        ),
        end='<|im_end|>\n',
        generate=True,
    ),
    dict(
        role='environment',
        begin=dict(
            with_name='<|im_start|>environment name={name}\n',
            without_name='<|im_start|>environment\n',
            name={
                'interpreter': '<|interpreter|>',
                'plugin': '<|plugin|>',
            },
        ),
        end='<|im_end|>\n',
    ),
]
system_prompt = """You are an AI language model employing iterative reasoning through defined actions during problem-solving, each encapsulated within specific XML tags:

- <restate>...</restate>
- <planning>...</planning>
- <recollect>...</recollect>
- <execution>...</execution>
- <review>...</review>
- <summarize>...</summarize>
- <conclude>...</conclude>

## Roles and Responsibilities

### \\<restate\\>
- **Objective**: Clearly articulate the problem conditions.
- **Instructions**: Summarize the key points of the problem succinctly.
- **Output Format**: Enclose your restatement within </restate> tags.

### \\<recollect\\>
- **Objective**: Recollect related technics that help solve the problem.
- **Instructions**: Recollect basic math skills, theorems, and techniques that may be used to solve the problem in the format of the list of relevant skills. You should elaborate the complete content of related technics so that you can skillfully apply them later. Use a list format to give your recollection of the relevant skills. If no specific skills or theorems can be thought about in the provided problem, respond with **[no skills]**.
- **Output Format**: Enclose your recollection within </recollect> tags.

### \\<planning\\>
- **Objective**: Break the problem into smaller, manageable parts.
- **Instructions**: Outline the individual components of the problem, record the updated plan after every step of execution. If a previous strategy fails, develop a new plan based on the insights gained.
- **Output Format**: Enclose your planning within </planning> tags.

### \\<execution\\>
- **Objective**: Implement one reasoning approach.
- **Instructions**: Carry out calculations or logical steps based on the current plans. Limit execution to a maximum of two steps before reviewing.
- **Output Format**: Enclose your execution within </execution> tags.

### \\<review\\>
- **Objective**: Verify the correctness of the reasoning process.
- **Instructions**: Assuming that there was an error in the previous implementation process, identify all possible problem areas and analyze them one by one. Record the error reason and any edge condition.
- **Output Format**: Enclose your review within <review> tags.

### \\<summarize\\>
- **Objective**: Document assumptions and intermediate results.
- **Instructions**: Gather all previously validated reasoning into a coherent log after completing one step of reasoning.
- **Output Format**: Enclose your summary within <summarize> tags.

### \\<conclude\\>
- **Objective**: Synthesize all previous execution paths and provide a final answer.
- **Instructions**: Formulate a comprehensive, professional, and formal final answer to the original question. Focus only on the correct reasoning paths identified during the problem-solving process. This conclusion should serve as a standalone, authoritative response to the initial query.
- **Output Format**: Enclose your conclusion within <conclude> tags.

## Process Flow

1. **Iteration Begins**: The iteration begins with the model using any action from the defined set based on its reasoning. Careful planning should be done before performing any other action.
2. **Recollect Relevant Skills**: Your job is to first analyze the problem and think about the possible basic math skills, theorems, and techniques that may be used to solve the problem.
3. **Critical Planning**: After each action, the model evaluates the results and determines the next appropriate action to take. If a strategy fails, develop a new plan based on the insights gained.
4. **Step-by-Step Execution**: After each planning session, the model executes one or two steps of the planned action, and records intermediate results to verify the accuracy of the current step.
5. **Review and Reflect**: The model conducts a thorough review after each execution step. This review process is treated as an independent verification of the work done. If errors are identified, reflect on the mistakes made and re-execute to correct these errors.
6. **conclude**: This cycle continues until the model synthesizes all validated reasoning and reaches a confident conclusion, using <conclude> to present the final, professional answer.

## Formatting Guidelines

- Clarity: Ensure each reasoning step and critique is easy to understand.
- Logical Progression: Each proposition should logically follow from previous ones, considering any critiques.
- Tags: Always encapsulate your output within the correct XML tags.
- Natural Language: Use detailed explanations in critiques to provide meaningful feedback.
- Verbosity: Strive to generate longer, more detailed responses to maximize the use of available tokens. Aim to use at least 70% of the maximum available token count in your responses.

Solve the giving problem with above formats in detailed reasoning process. Provide answers in the same language as the user asking the question, repeat the final answer in <conclude>..</conclude> using a '\\boxed{{}}' without any units, you have [[{token_limit}]] tokens to complete the answer."""


class InternThinkerAggregator(DefaultAggregator):
    def aggregate(self, messages, name, parser=None, system_instruction=None):
        _message, cur_role = [], None
        for msg in super().aggregate(messages, name, parser, system_instruction):
            if cur_role == msg['role']:
                _message[-1]['content'] += msg['content']
            else:
                _message.append(msg)
                cur_role = msg['role']
        return _message


class InternThinkerAgent(Agent):
    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        template: Union[str, dict, List[dict]] = None,
        aggregator: Dict = dict(type=InternThinkerAggregator),
        hooks: Callable = None,
        max_turn: int = 2,
        **kwargs,
    ):
        super().__init__(hooks=hooks, **kwargs)
        self.agent = Agent(
            llm=llm,
            template=template,
            aggregator=aggregator,
            hooks=hooks,
            **kwargs,
        )
        self.max_turn = max_turn

    def forward(self, *message, session_id=0, **kwargs):
        message = self.agent(*message, session_id=session_id, **kwargs).model_copy()
        message.sender = self.name
        message.content = message.content.replace('<conclude>', '<conclude_draft>').replace('</conclude>', '</conclude_draft>')
        for i in range(self.max_turn):
            append = self.agent(AgentMessage(sender=self.agent.name, content='\n\n'), session_id=session_id, **kwargs)
            if i < self.max_turn - 1:
                append.content = append.content.replace('<conclude>', '<conclude_draft>').replace('</conclude>', '</conclude_draft>')
            message.content += '\n\n' + append.content
        return message


class StreamingInternThinkerAgent(StreamingAgent):
    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        template: Union[str, dict, List[dict]] = None,
        hooks: Callable = None,
        max_turn: int = 2,
        aggregator: Dict = dict(type=InternThinkerAggregator),
        **kwargs,
    ):
        super().__init__(hooks=hooks, **kwargs)
        self.agent = StreamingAgent(llm=llm, template=template, hooks=hooks, aggregator=aggregator, **kwargs)
        self.max_turn = max_turn

    def forward(self, *message, session_id=0, **kwargs):
        for response_message in self.agent(*message, session_id=session_id, **kwargs):

            inner_message = response_message.model_copy(
                update=dict(
                    content=response_message.content.replace('<conclude>', '<conclude_draft>').replace(
                        '</conclude>', '</conclude_draft>'
                    )
                )
            )
            yield inner_message
        response_message = inner_message.model_copy()
        for i in range(self.max_turn):
            for inner_message in self.agent(
                AgentMessage(sender=self.agent.name, content='\n\n'), session_id=session_id, **kwargs
            ):

                if i < self.max_turn - 1:
                    inner_message.content = inner_message.content.replace('<conclude>', '<conclude_draft>').replace(
                        '</conclude>', '</conclude_draft>'
                    )
                yield response_message.model_copy(
                    update=dict(content=response_message.content + '\n\n' + inner_message.content)
                )
            response_message = response_message.model_copy(
                update=dict(content=response_message.content + '\n\n' + inner_message.content)
            )
        return response_message


class AsyncStreamingInternThinkerAgent(AsyncStreamingAgent):
    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        template: Union[str, dict, List[dict]] = None,
        hooks: Callable = None,
        max_turn: int = 2,
        aggregator: Dict = dict(type=InternThinkerAggregator),
        **kwargs,
    ):
        super().__init__(llm=llm, template=template, hooks=hooks, **kwargs)
        self.agent = AsyncStreamingAgent(llm=llm, template=template, hooks=hooks, aggregator=aggregator, **kwargs)
        self.max_turn = max_turn

    async def forward(self, *message, session_id=0, **kwargs):
        response_message = AgentMessage(sender=self.name, content='')
        async for response_message in self.agent(*message, session_id=session_id, **kwargs):
            inner_message = response_message.model_copy(
                update=dict(
                    content=response_message.content.replace('<conclude>', '<conclude_draft>').replace(
                        '</conclude>', '</conclude_draft>'
                    )
                )
            )
            yield inner_message
        response_message = inner_message.model_copy()
        for i in range(self.max_turn):
            async for inner_message in self.agent(
                AgentMessage(sender=self.agent.name, content='\n\n'), session_id=session_id, **kwargs
            ):
                if i < self.max_turn - 1:
                    inner_message.content = inner_message.content.replace('<conclude>', '<conclude_draft>').replace(
                        '</conclude>', '</conclude_draft>'
                    )

                yield response_message.model_copy(
                    update=dict(content=response_message.content + '\n\n' + inner_message.content)
                )
            response_message = response_message.model_copy(
                update=dict(content=response_message.content + '\n\n' + inner_message.content)
            )


if __name__ == '__main__':
    llm = AsyncLMDeployClient(
        url='http://localhost:23333',
        model_name='qwen_o1',
        meta_template=INTERNLM2_META,
        max_new_tokens=8192,
        stop_words=['<|im_end|>'],
    )
    agent = AsyncStreamingInternThinkerAgent(llm=llm, template=system_prompt)
    message = AgentMessage(sender='user', content='What is the sum of 2 and 3?')
    import asyncio

    loop = asyncio.get_event_loop()

    async def streaming_agent(input):
        response = AgentMessage(sender='assistant', content='')
        async for response in agent(input):
            print(response.content)
        return response

    response = loop.run_until_complete(streaming_agent(message))
    print(response)
    loop.close()
