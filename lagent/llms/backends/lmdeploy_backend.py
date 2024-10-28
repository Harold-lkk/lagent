import asyncio
from dataclasses import asdict
from typing import Callable, List, Literal, Optional, Union

import aiohttp
from typing_extensions import NotRequired, TypedDict

from lagent.llms.backends.base_backend import AsyncMixin, LocalBackend
from lagent.utils.util import filter_suffix


class LMDeployTurbomindEnginebConfig(TypedDict, total=False):
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


class LMDepolyChatTemplateConfig(TypedDict):
    model_name: str
    system: Optional[str] = None
    meta_instruction: Optional[str] = None
    eosys: Optional[str] = None
    user: Optional[str] = None
    eoh: Optional[str] = None
    assistant: Optional[str] = None
    eoa: Optional[str] = None
    separator: Optional[str] = None
    capability: Optional[Literal['completion', 'infilling', 'chat',
                                 'python']] = None
    stop_words: Optional[List[str]] = None


class LMDeployBackendConfig(TypedDict):
    model_name: Optional[str]
    backend_config: LMDeployTurbomindEnginebConfig
    chat_template_config: NotRequired[LMDepolyChatTemplateConfig]


class LMDeployBackend(LocalBackend):

    def init_client(self, model: str, backend_config: LMDeployBackendConfig,
                    gen_params) -> Callable:
        from lmdeploy import ChatTemplateConfig, TurbomindEngineConfig, pipeline
        model_name = backend_config.pop('model_name', None)
        chat_template_config = ChatTemplateConfig(
            model_name=model_name) if model_name else None

        self.model = pipeline(
            model_path=model,
            backend_config=TurbomindEngineConfig(**backend_config),
            chat_template_config=chat_template_config,
        )

    def _completion(
        self,
        inputs: Union[str, List[str]],
        **gen_params,
    ):
        """Return the chat completions in non-stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
        Returns:
            (a list of/batched) text/chat completion
        """
        from lmdeploy.messages import GenerationConfig
        gen_config = GenerationConfig(**gen_params)
        response = self.model.batch_infer(
            inputs, gen_config=gen_config, do_preprocess=False)
        return response


class AsyncLMDeployBackend(Asyncixin, LMDeployBackend):
    """

    Args:
        path (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download
                        from ii) and iii).
                    - ii) The model_id of a lmdeploy-quantized model hosted
                        inside a model repo on huggingface.co, such as
                        "InternLM/internlm-chat-20b-4bit",
                        "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on.
        tp (int): tensor parallel
        pipeline_cfg (dict): config of pipeline
    """

    async def completion(
        self,
        inputs: Union[str, List[str]],
        **kwargs,
    ):
        """Return the chat completions in non-stream mode.

        Args:
            inputs (Union[str, List[str]]): input texts to be completed.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            skip_special_tokens (bool): Whether or not to remove special tokens
                in the decoding. Default to be False.
        Returns:
            (a list of/batched) text/chat completion
        """
        from lmdeploy.messages import GenerationConfig, Response

        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        if session_ids is None:
            session_ids = list(range(len(inputs)))
        elif isinstance(session_ids, (int, str)):
            session_ids = [session_ids]
        assert len(inputs) == len(session_ids)

        prompt = inputs
        gen_params = self.update_gen_params(**kwargs)
        gen_config = GenerationConfig(**gen_params)

        async def _inner_generate(uid, text):
            resp = Response('', 0, 0, uid)
            async for out in self.model.generate(
                    text,
                    uid,
                    gen_config,
                    stream_response=True,
                    sequence_start=True,
                    sequence_end=True,
                    do_preprocess=do_preprocess,
                    **kwargs,
            ):
                resp.text += out.response
                resp.generate_token_len = out.generate_token_len
                resp.input_token_len = out.input_token_len
                resp.finish_reason = out.finish_reason
                if out.token_ids:
                    resp.token_ids.extend(out.token_ids)
                if out.logprobs:
                    if resp.logprobs is None:
                        resp.logprobs = []
                    resp.logprobs.extend(out.logprobs)
            return resp

        response = await asyncio.gather(*[
            _inner_generate(sid, inp) for sid, inp in zip(session_ids, prompt)
        ])
        texts = [resp.text for resp in response]
        # remove stop_words
        texts = filter_suffix(texts, self.gen_params.get('stop_words'))
        for resp, text in zip(response, texts):
            resp.text = text
        if batched:
            return [asdict(resp)
                    for resp in response] if return_dict else texts
        return asdict(response[0]) if return_dict else texts[0]
