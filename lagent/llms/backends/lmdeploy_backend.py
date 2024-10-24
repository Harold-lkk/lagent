import asyncio
from dataclasses import asdict
from typing import Callable, List, Optional, Union

import aiohttp

from lagent.llms.backends.base_backend import LocalBackend, AsyncMixin
from lagent.utils.util import filter_suffix


class LMDeployBackend(LocalBackend):
    def init_client(self, model, backend_config) -> Callable:
        from lmdeploy import TurbomindEngineConfig, pipeline

        self.model = pipeline(
            model_path=model,
            backend_config=TurbomindEngineConfig(**backend_config),
            )

    def completion(
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
        from lmdeploy.messages import GenerationConfig

        batched = True
        if isinstance(inputs, str):
            inputs = [inputs]
            batched = False
        prompt = inputs
        gen_params = self.update_gen_params(**kwargs)
        gen_config = GenerationConfig(**gen_params)
        response = self.model.batch_infer(
            prompt, gen_config=gen_config, do_preprocess=False)
        texts = [resp.text for resp in response]
        # remove stop_words
        texts = filter_suffix(texts, self.gen_params.get('stop_words'))
        for resp, text in zip(response, texts):
            resp.text = text
        if batched:
            return [asdict(resp)
                    for resp in response] if return_dict else texts
        return asdict(response[0]) if return_dict else texts[0]


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
#         session_ids: Union[int, List[int]] = None,
#         do_preprocess: bool = None,
#         skip_special_tokens: bool = False,
#         return_dict: bool = False,
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
#         gen_config = GenerationConfig(
#             skip_special_tokens=skip_special_tokens, **gen_params)

#         async def _inner_generate(uid, text):
#             resp = Response('', 0, 0, uid)
#             async for out in self.model.generate(
#                     text,
#                     uid,
#                     gen_config,
#                     stream_response=True,
#                     sequence_start=True,
#                     sequence_end=True,
#                     do_preprocess=do_preprocess,
#                     **kwargs,
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

#         response = await asyncio.gather(*[
#             _inner_generate(sid, inp) for sid, inp in zip(session_ids, prompt)
#         ])
#         texts = [resp.text for resp in response]
#         # remove stop_words
#         texts = filter_suffix(texts, self.gen_params.get('stop_words'))
#         for resp, text in zip(response, texts):
#             resp.text = text
#         if batched:
#             return [asdict(resp)
#                     for resp in response] if return_dict else texts
#         return asdict(response[0]) if return_dict else texts[0]
