import os
from typing import Dict, List, Union

from lagent.llms.backends.base_backend import AsyncMixin, LocalBackend, RemoteBackend
from lagent.llms.backends.lmdeploy_backend import LMDeployBackend


class LLM:
    _backends = {
        'local': {
            'lmdeploy': LMDeployBackend,
            # Add other local backends here
        },
        'remote': {
            #     'http': HTTPBackend,
            #     # Add other remote backends here
        },
    }

    _default_backend_config = {
        'lmdeploy': {},
    }

    # Class-level dictionary to store instances for singleton
    _instances = {}

    def __new__(cls,
                *,
                backend,
                model=None,
                base_url=None,
                api_key=None,
                proxy=None,
                gen_params=None,
                backend_config=None,
                singleton=True):
        """Control instance creation based on whether singleton is enabled or not."""
        # If singleton is enabled, check if the instance already exists
        if singleton:
            # Create a unique key based on parameters
            instance_key = (
                backend, model, base_url, api_key,
                frozenset(gen_params.items()) if gen_params else None,
                frozenset(backend_config.items()) if backend_config else None)

            # Return the cached instance if it exists
            if instance_key in cls._instances:
                return cls._instances[instance_key]

            # Create a new instance and cache it
            instance = super().__new__(cls)
            cls._instances[instance_key] = instance
            return instance
        else:
            # If singleton is disabled, always create a new instance
            return super().__new__(cls)

    def __init__(self,
                 *,
                 backend,
                 model=None,
                 base_url=None,
                 api_key=None,
                 gen_params=None,
                 backend_config=None,
                 singleton=True,
                 proxy=None,
                 role_map: List[Dict] = None,
                 return_type='str'):
        """Initialize the appropriate backend based on the backend type."""
        # Avoid re-initialization if already initialized (especially in singleton mode)
        if hasattr(self, 'initialized') and self.initialized:
            return

        # If no argument is passed, fall back to class-level variables
        self.base_url = base_url or self.base_url
        self.api_key = api_key or self.api_key
        self.model = model or self.model.get(backend)
        self.role_map = role_map
        # Merge backend config with defaults
        _backend_config = {
            **self._default_backend_config.get(backend, {}),
            **(backend_config or {})
        }

        # Initialize the backend based on whether it's local or remote
        self.backend = self._initialize_backend(
            backend=backend,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            gen_params=gen_params,
            backend_config=_backend_config)

        # Mark the instance as initialized to prevent reinitialization
        self.initialized = True

    def _initialize_backend(
            self, backend, model, base_url, api_key, gen_params,
            backend_config) -> Union[LocalBackend, RemoteBackend]:
        """Helper function to initialize the backend based on its type."""
        if backend in self._backends['local']:
            return self._backends['local'][backend](
                model=model,
                gen_params=gen_params,
                backend_config=backend_config,
            )
        elif backend in self._backends['remote']:
            return self._backends['remote'][backend](
                model=model,
                base_url=base_url,
                api_key=api_key,
                gen_params=gen_params,
                backend_config=backend_config,
            )
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Supported backends: {self._backends.get('local', {}).keys() + self._backends.get('remote', {}).keys()}"
            )

    def chat_completion(self, inputs: Union[str, List[str]],
                        **gen_params) -> str:
        """Generate results given a str (or list of) inputs.

        Args:
            inputs (Union[str, List[str]]):
            gen_params (dict): The input params for generation.

        Returns:
            Union[str, List[str]]: A (list of) generated strings.

        eg.
            batched = True
            if isinstance(inputs, str):
                inputs = [inputs]
                batched = False
            response = ['']
            if batched:
                return response
            return response[0]
        """
        if self.role_map:
            inputs = self.map_message(inputs)
        if self.backend.whether_support_model(self.model):
            return self.backend.chat_completion(inputs, **gen_params)
        else:
            return self.backend.completion(
                self.chat_template(inputs), **gen_params)

    def completion(self, inputs: str, **gen_params) -> List[str]:
        """Generate results as streaming given a str inputs.

        Args:
            inputs (str):
            gen_params (dict): The input params for generation.

        Returns:
            str: A generated string.
        """
        return self.backend.completion(inputs, **gen_params)


class AsyncLLM(AsyncMixin, LLM):

    async def chat_completion(self, inputs: Union[str, List[str]],
                              **gen_params) -> str:
        return await self.backend.chat_completion(inputs, **gen_params)

    async def completion(self, inputs: str, **gen_params) -> List[str]:
        return await self.backend.completion(inputs, **gen_params)


class InternLM(LLM):
    base_url = 'https://puyu.openxlab.org.cn/puyu/api/v1'
    api_key = os.getenv('INTERNLM_API_KEY')

    model = dict(
        local='internlm/internlm-chat-7b',
        remote='internlm2-chat-7b',
    )
    _default_backend_config = {
        'lmdeploy': {
            'model_name': 'internlm2-chat',
        },
        'api': {
            'Content-Type': 'application/json',
        }
    }

    chat_template = 'str'


class Qwen(LLM):
    base_url = 'https://puyu.openxlab.org.cn/puyu/api/v1'
    api_key = os.getenv('INTERNLM_API_KEY')

    model = dict(
        local='Qwen/Qwen2.5-7B-Instruct',
        remote='qwen',
    )
    _default_backend_config = {
        'lmdeploy': {
            'model_name': 'qwen',
        },
        'api': {
            'Content-Type': 'application/json',
        }
    }

    chat_template = 'str'


if __name__ == '__main__':

    # internlm = InternLM(backend='transformers')  # from huggingface
    qwen = Qwen(
        backend='lmdeploy',
        model='/fs-computility/llm/shared/llm_qwen/Qwen2.5-7B-Instruct'
    )  # from local

    # internlm = InternLM(backend='api', model='internlm2-chat-20b',)  # for puyu official api
    # internlm = InternLM(backend='api', base_url='localhost:2333', model='internlm2-chat-20b', api_key=os.getenv('custom_key'))  # for proxy server
    response = qwen.chat_completion([dict(role='user',
                                          text='hello')])  # chat_completion
    print(response)
