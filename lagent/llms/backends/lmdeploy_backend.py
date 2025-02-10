import asyncio
from dataclasses import asdict
from typing import Callable, List, Literal, Optional, Union

import aiohttp
import jinja2
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion
from typing_extensions import NotRequired, TypedDict

from lagent.llms.schema import GenerateParams


class LMDeployBackendConfig(TypedDict, total=False):
    model_name: NotRequired[str]
    model_format: NotRequired[str]
    tp: int = 1
    session_len: NotRequired[int]
    max_batch_size: int
    cache_max_entry_count: float
    cache_block_seq_len: int
    enable_prefix_caching: bool
    quant_policy: int
    rope_scaling_factor: float
    use_logn_attn: bool = False
    download_dir: NotRequired[str]
    revision: NotRequired[str]
    max_prefill_token_num: int
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1


class LMDeployBackend:

    def __init__(self,
                 model: str,
                 backend_config: LMDeployBackendConfig,
                 gen_params: GenerateParams,
                 chat_template: Optional[str] = None) -> None:
        from lmdeploy import TurbomindEngineConfig, pipeline, version_info
        self.lmdeploy_version = version_info
        self.model = model
        self.client = pipeline(
            model_path=model,
            backend_config=TurbomindEngineConfig(**backend_config),
        )
        self.backend_config = backend_config
        self.gen_params = gen_params
        self.chat_template = jinja2.Template(chat_template)

    def complete(self,
                 inputs: Union[str, List[str]],
                 **gen_params)-> Union[Completion, List[Completion]]:
        """Return the chat completions in non-stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
        Returns:
            (a list of/batched) text/chat completion
        """
        from lmdeploy.messages import GenerationConfig
        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs
        gen_params = self.update_gen_params(**gen_params)
        gen_config = GenerationConfig(**gen_params)
        response = self.client.batch_infer(
            prompt, gen_config=gen_config, do_preprocess=False)
        #     id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
        #     object: str = 'chat.completion.chunk'
        #     created: int = Field(default_factory=lambda: int(time.time()))
        # remove stop_words
        import time

        import shortuuid
        completion_response = []
        for resp in response:
            completion_response.append(Completion(
                id=f'cmpl-{shortuuid.random()}',
                object='text_completion',
                created=int(time.time()),
                model=self.backend_config.get('model_name') or self.model,
                choices=[
                    {
                        'finish_reason': resp.finish_reason,
                        'index': 0,
                        'text': resp.text,
                        'logprobs':
                            {
                            'token_logprobs': resp.logprobs
                            'tokens': [str(t_id) for t_id in resp.token_ids],
                            'top_logprobs': resp.logprobs,
                        }
                    }
                ],
                usage={
                    'completion_tokens': resp.generate_token_len,
                    'prompt_tokens': resp.input_token_len,
                    'total_tokens': resp.generate_token_len + resp.input_token_len
                }

            ))
        if batched:
            return completion_response
        return completion_response[0]

    def chat_complete(self, inputs: Union[str, List[str]], **kwargs) -> Union[ChatCompletion, List[ChatCompletion]]:
        assert isinstance(inputs, list)
        is_batched = isinstance(inputs[0], list)
        if not is_batched:
            inputs = [inputs]
        assert self.chat_template is not None

        inputs = [self.chat_template.render(messages=input) for input in inputs]
        response = self.complete(inputs, **kwargs)
        ## convert Complete to ChatCompletion
        chat_completion = []
        for resp in response:
            log_probs = []
            # for choice in resp.choices:
            #     if choice.logprobs:
            #         log_probs.append({
            #             'content': [
            #                 {
            #                     'token': token,
            #                     'logprob': logprob,
            #                     'top_logprobs': top_logprobs
            #                 }
            #                 for token, logprob, top_logprobs in zip(
            #                     choice.logprobs.tokens, choice.logprobs.token_logprobs or [None] * len(choice.logprobs.tokens), choice.logprobs.top_logprobs or [None] * len(choice.logprobs.tokens))
            #             ]
            chat_completion.append(ChatCompletion(
                id=resp.id,
                object='chat.completion',
                created=resp.created,
                model=resp.model,
                choices=[
                    {
                        'finish_reason': choice.finish_reason,
                        'index': choice.index,
                        'message': {'content': choice.text},
                        'logprobs': {
                            'content': [
                                {
                                    'token': token,
                                    'logprob': logprob,
                                    'top_logprobs': top_logprobs
                                }
                                for token, logprob, top_logprobs in zip(
                                    choice.logprobs.tokens, choice.logprobs.token_logprobs or [None] * len(choice.logprobs.tokens), choice.logprobs.top_logprobs or [None] * len(choice.logprobs.tokens))
                            ]
                        }
                    }
                    for choice in resp.choices
                ],
                usage=resp.usage
            ))

    def stream_complete(self, inputs: str, **kwargs):
        pass

    def stream_chat_complete(self, inputs: str, **kwargs):
        pass

    def update_gen_params(self, **kwargs):
        gen_params = self.gen_params.copy()
        gen_params.update(kwargs)
        do_sample = gen_params.pop('do_sample', None)
        if do_sample is not None and self.lmdeploy_version < (0, 6, 0):
            raise RuntimeError(
                '`do_sample` parameter is not supported by lmdeploy until '
                f'v0.6.0, but currently using lmdeloy {".".join(self.lmdeploy_version)}')
        if self.lmdeploy_version >= (0, 6, 0):
            if do_sample is None:
                do_sample = gen_params['top_k'] > 1 or gen_params[
                    'temperature'] > 0
            gen_params.update(do_sample=do_sample)

        max_tokens = gen_params.pop('max_tokens', None)
        if max_tokens is not None:
            gen_params['max_new_tokens'] = max_tokens
        return gen_params


# class AsyncLMDeployBackend(Asyncixin, LMDeployBackend):
#     """

#     Args:
#         path (str): The path to the model.
#             It could be one of the following options:
#                     - i) A local directory path of a turbomind model which is
#                         converted by `lmdeploy convert` command or download
#                         from ii) and iii).
#                     - ii) The model_id of a lmdeploy-quantized model hosted
#                         inside a model repo on huggingface.co, such as
#                         "InternLM/internlm-chat-20b-4bit",
#                         "lmdeploy/llama2-chat-70b-4bit", etc.
#                     - iii) The model_id of a model hosted inside a model repo
#                         on huggingface.co, such as "internlm/internlm-chat-7b",
#                         "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
#                         and so on.
#         model_name (str): needed when model_path is a pytorch model on
#             huggingface.co, such as "internlm-chat-7b",
#             "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
#         tp (int): tensor parallel
#         pipeline_cfg (dict): config of pipeline
#     """

#     async def generate(
#         self,
#         inputs: Union[str, List[str]],
#         **kwargs,
#     ):
#         """Return the chat completions in non-stream mode.

#         Args:
#             inputs (Union[str, List[str]]): input texts to be completed.
#             do_preprocess (bool): whether pre-process the messages. Default to
#                 True, which means chat_template will be applied.
#             skip_special_tokens (bool): Whether or not to remove special tokens
#                 in the decoding. Default to be False.
#         Returns:
#             (a list of/batched) text/chat completion
#         """
#         from lmdeploy.messages import GenerationConfig, Response

#         batched = True
#         if isinstance(inputs, str):
#             inputs = [inputs]
#             batched = False
#         if session_ids is None:
#             session_ids = list(range(len(inputs)))
#         elif isinstance(session_ids, (int, str)):
#             session_ids = [session_ids]
#         assert len(inputs) == len(session_ids)

#         prompt = inputs
#         gen_params = self.update_gen_params(**kwargs)
#         gen_config = GenerationConfig(**gen_params)

#         async def _inner_generate(uid, text):
#             resp = Response('', 0, 0, uid)
#             async for out in self.client.generate(
#                 text,
#                 uid,
#                 gen_config,
#                 stream_response=True,
#                 sequence_start=True,
#                 sequence_end=True,
#                 do_preprocess=do_preprocess,
#                 **kwargs,
#             ):
#                 resp.text += out.response
#                 resp.generate_token_len = out.generate_token_len
#                 resp.input_token_len = out.input_token_len
#                 resp.finish_reason = out.finish_reason
#                 if out.token_ids:
#                     resp.token_ids.extend(out.token_ids)
#                 if out.logprobs:
#                     if resp.logprobs is None:
#                         resp.logprobs = []
#                     resp.logprobs.extend(out.logprobs)
#             return resp

#         response = await asyncio.gather(*[_inner_generate(sid, inp) for sid, inp in zip(session_ids, prompt)])
#         texts = [resp.text for resp in response]
#         # remove stop_words
#         texts = filter_suffix(texts, self.gen_params.get('stop_words'))
#         for resp, text in zip(response, texts):
#             resp.text = text
#         if batched:
#             return [asdict(resp) for resp in response] if return_dict else texts
#         return asdict(response[0]) if return_dict else texts[0]


if __name__ == '__main__':
    backend = LMDeployBackend(
        model='/fs-computility/llm/shared/llm_qwen/Qwen2.5-0.5B-Instruct',
        backend_config={
            'tp': 1,
        },
        gen_params=GenerateParams(
            n=1,
            temperature=1,
            max_tokens=256,
            top_k=40,
            logprobs=100,
        ),
        chat_template="""{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n""",
    )
    print(backend.chat_complete([dict(role='user', content='hello')]))
