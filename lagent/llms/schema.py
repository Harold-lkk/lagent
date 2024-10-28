import time
from typing import TYPE_CHECKING, Annotated, Callable, Dict, List, Literal, Optional, Union

import shortuuid
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    import torch

LogitsProcessor = Callable[[List[int], 'torch.Tensor'], 'torch.Tensor']


class GenerateParams(TypedDict, total=False):
    temperature: float
    top_p: float
    top_k: int
    max_tokens: Optional[int]
    repetition_penalty: float
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[List[int]]
    do_sample: bool
    skip_special_tokens: bool
    n: int
    best_of: NotRequired[int]
    presence_penalty: NotRequired[float]
    frequency_penalty: NotRequired[float]
    min_p: float
    seed: Optional[int]
    use_beam_search: bool
    length_penalty: float
    early_stopping: Union[bool, str]
    include_stop_str_in_output: bool
    ignore_eos: bool
    min_tokens: int
    logprobs: Optional[int]
    prompt_logprobs: Optional[int]
    detokenize: bool
    spaces_between_special_tokens: bool
    logits_processors: Optional[List[LogitsProcessor]]
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]]


class UsageInfo(BaseModel):
    """Usage information."""
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion response choices."""
    index: int
    message: List[Dict[str, str]]
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls']] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """Delta messages."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    """Chat completion response stream choice."""
    index: int
    delta: DeltaMessage
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal['stop', 'length']] = None


class ChatCompletionStreamResponse(BaseModel):
    """Chat completion stream response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion.chunk'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
