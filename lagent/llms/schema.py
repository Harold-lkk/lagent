from typing import TYPE_CHECKING, Annotated, Callable, List, Optional, Union

from pydantic import Field
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
