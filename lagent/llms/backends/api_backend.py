import asyncio
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
from typing import Dict, List, Optional, Union

import aiohttp
import requests

from lagent.llms.backends.base_backend import AsyncMixin, RemoteBackend


class APIBackend(RemoteBackend):

    def chat_completion(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        **gen_params,
    ) -> Union[str, List[str]]:
        """Generate responses given the contexts.

        Args:
            inputs (Union[List[dict], List[List[dict]]]): a list of messages
                or list of lists of messages
            gen_params: additional generation configuration

        Returns:
            Union[str, List[str]]: generated string(s)
        """
        assert isinstance(inputs, list)
        if 'max_tokens' in gen_params:
            raise NotImplementedError('unsupported parameter: max_tokens')
        gen_params = {**self.gen_params, **gen_params}
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [
                executor.submit(self._chat, **gen_params)
                for messages in ([inputs] if isinstance(inputs[0], dict) else inputs)
            ]
        ret = [task.result() for task in tasks]
        return ret[0] if isinstance(inputs[0], dict) else ret

    async def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates.

        Args:
            messages (List[dict]): a list of prompt dictionaries
            gen_params: additional generation configuration

        Returns:
            str: The generated string.
        """
        assert isinstance(messages, list)

        header, data = self.generate_request_data(
            model_type=self.model_type,
            messages=messages,
            gen_params=gen_params,
            json_mode=self.json_mode,
        )

        max_num_retries = 0
        while max_num_retries < self.retry:
            if len(self.invalid_keys) == len(self.keys):
                raise RuntimeError('All keys have insufficient quota.')

            # find the next valid key
            while True:
                self.key_ctr += 1
                if self.key_ctr == len(self.keys):
                    self.key_ctr = 0

                if self.keys[self.key_ctr] not in self.invalid_keys:
                    break

            key = self.keys[self.key_ctr]
            header['Authorization'] = f'Bearer {key}'

            if self.orgs:
                self.org_ctr += 1
                if self.org_ctr == len(self.orgs):
                    self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            response = dict()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.url,
                        headers=header,
                        json=data,
                        proxy=self.proxies.get('https', self.proxies.get('http')),
                    ) as resp:
                        response = await resp.json()
                        return response['choices'][0]['message']['content'].strip()
            except aiohttp.ClientConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            except aiohttp.ClientResponseError as e:
                self.logger.error('Response error, got ' + str(e))
                continue
            except json.JSONDecodeError:
                self.logger.error('JsonDecode error, got ' + (await resp.text(errors='replace')))
                continue
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    self.logger.error('Find error message in response: ' + str(response['error']))
            except Exception as error:
                self.logger.error(str(error))
            max_num_retries += 1

        raise RuntimeError(
            'Calling OpenAI failed after retrying for ' f'{max_num_retries} times. Check the logs for ' 'details.'
        )

    def generate_request_data(
        self,
        model_type,
        messages,
        gen_params,
    ):
        """
        Generates the request data for different model types.

        Args:
            model_type (str): The type of the model (e.g., 'gpt', 'internlm', 'qwen').
            messages (list): The list of messages to be sent to the model.
            gen_params (dict): The generation parameters.
            json_mode (bool): Flag to determine if the response format should be JSON.

        Returns:
            tuple: A tuple containing the header and the request data.
        """
        # Copy generation parameters to avoid modifying the original dictionary
        gen_params = gen_params.copy()

        # Hold out 100 tokens due to potential errors in token calculation
        max_tokens = min(gen_params.pop('max_new_tokens'), 4096)
        if max_tokens <= 0:
            return '', ''

        # Initialize the header
        header = {
            'content-type': 'application/json',
        }

        # Common parameters processing
        gen_params['max_tokens'] = max_tokens
        if 'stop_words' in gen_params:
            gen_params['stop'] = gen_params.pop('stop_words')
        if 'repetition_penalty' in gen_params:
            gen_params['frequency_penalty'] = gen_params.pop('repetition_penalty')

        # Model-specific processing
        data = {}
        if model_type.lower().startswith('gpt'):
            if 'top_k' in gen_params:
                warnings.warn(
                    '`top_k` parameter is deprecated in OpenAI APIs.',
                    DeprecationWarning,
                )
                gen_params.pop('top_k')
            gen_params.pop('skip_special_tokens', None)
            gen_params.pop('session_id', None)
            data = {'model': model_type, 'messages': messages, 'n': 1, **gen_params}
            if json_mode:
                data['response_format'] = {'type': 'json_object'}
        elif model_type.lower().startswith('internlm'):
            data = {'model': model_type, 'messages': messages, 'n': 1, **gen_params}
            if json_mode:
                data['response_format'] = {'type': 'json_object'}
        elif model_type.lower().startswith('qwen'):
            header['X-DashScope-SSE'] = 'enable'
            gen_params.pop('skip_special_tokens', None)
            gen_params.pop('session_id', None)
            if 'frequency_penalty' in gen_params:
                gen_params['repetition_penalty'] = gen_params.pop('frequency_penalty')
            gen_params['result_format'] = 'message'
            data = {
                'model': model_type,
                'input': {'messages': messages},
                'parameters': {**gen_params},
            }
        else:
            raise NotImplementedError(f'Model type {model_type} is not supported')

        return header, data

    def tokenize(self, prompt: str) -> list:
        """Tokenize the input prompt.

        Args:
            prompt (str): Input string.

        Returns:
            list: token ids
        """
        import tiktoken

        self.tiktoken = tiktoken
        enc = self.tiktoken.encoding_for_model(self.model_type)
        return enc.encode(prompt)
