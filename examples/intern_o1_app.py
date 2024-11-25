from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union
import os
import shortuuid
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from lagent.agents.intern_o1 import INTERNLM2_META, AsyncStreamingInternThinkerAgent, system_prompt
from lagent.llms import AsyncLMDeployClient


class GenerateRequest(BaseModel):
    """Generate request."""

    prompt: Union[str, List[Dict[str, Any]]]
    image_url: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])
    session_id: int = -1
    interactive_mode: bool = False
    stream: bool = True
    stop: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])
    request_output_len: Optional[int] = 8192  # noqa
    top_p: float = 0.8
    top_k: int = 40
    temperature: float = 0.7
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    skip_special_tokens: Optional[bool] = True
    cancel: Optional[bool] = False  # cancel a responding request
    adapter_name: Optional[str] = Field(default=None, examples=[None])
    model: Optional[str] = Field(default=None, examples=[None])


class GenerateResponse(BaseModel):
    """Generate response."""

    text: str
    tokens: int = 0
    input_tokens: int = 0
    history_tokens: int = 0
    finish_reason: Optional[Literal['stop', 'length']] = None


app = FastAPI(docs_url='/')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # 允许所有来源的请求，生产环境下请设置特定域名
    allow_credentials=True,
    allow_methods=['*'],  # 允许所有方法（GET, POST 等）
    allow_headers=['*'],  # 允许所有请求头
)

agent = AsyncStreamingInternThinkerAgent(
    llm=AsyncLMDeployClient(
        url=os.getenv('MODEL_URL', 'http://localhost:39999'),
        model_name=os.getenv('MODEL_NAME', 'qwen_o1'),
        meta_template=INTERNLM2_META,
        max_new_tokens=8192,
        stop_words=['<|im_end|>'],
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.0
    ),
    template=system_prompt,
)


@app.post('/v1/chat/interactive')
async def chat_interactive_v1(request: GenerateRequest, raw_request: Request = None):
    # 流式生成器函数
    async def stream_response(
        prompt: str,
        session_id: int,
    ) -> AsyncGenerator[bytes, None]:
        previous_content = ''  # 存储上一次的内容
        async for response in agent(prompt, session_id=session_id):
            current_content = response.content
            # 找到新增部分
            incremental_content = current_content[len(previous_content) :]
            previous_content = current_content
            if incremental_content:  # 如果有新增部分，才返回
                chunk = GenerateResponse(
                    text=incremental_content,
                )
                data = chunk.model_dump_json()
                yield f'{data}\n'

    if not request.interactive_mode:
        agent.reset(session_id=request.session_id, recursive=True)
        return JSONResponse(content={'message': 'Interactive mode is off.'})
    # 如果是交互模式
    if isinstance(request.prompt, str):
        # 调用代理生成响应
        return StreamingResponse(stream_response(request.prompt, request.session_id), media_type='text/event-stream')
    elif isinstance(request.prompt, list):
        # 加载历史上下文
        inputs = {'memory': [], 'agent.memory': []}
        for p in request.prompt[:-1]:
            if p['role'] == 'user':
                inputs['memory'].append(p)
                inputs['agent.memory'].append(p)
            else:
                inputs['memory'].append(dict(sender=agent.name, content=p['content']))
                inputs['agent.memory'].append(
                    dict(sender=agent.agent.name, content=p['content'].replace('<conclude_draft>', '<conclude>'))
                )

        agent.load_state_dict(inputs, session_id=request.session_id)
        # 返回最新用户输入的处理结果
        return StreamingResponse(
            stream_response(request.prompt[-1]['content'], request.session_id),
            media_type='text/event-stream',
        )
    else:
        raise HTTPException(status_code=400, detail='Invalid prompt format.')


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=39997,
        log_level='info',
        ssl_keyfile=None,
        ssl_certfile=None,
    )
