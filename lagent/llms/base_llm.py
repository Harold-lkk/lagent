from copy import copy
from typing import List, Union


class BaseLLM:

    # _backends = {
    #     'lmdeploy': LMDeployBackend,
    #     'vllm': VLLMBackend,
    #     'hf': HFBackend,
    #     'api': HTTPBackend,
    # }
    # default backend config put in backend_config
    _default_backend_config = {
        'lmdeploy': {},
    }

    _instances: dict = {}

    def __new__(cls, backend=None, **kwargs):

        if backend is not None and backend not in cls._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(cls._backends.keys())}')

        # concatenate the arguments to a unique key for determining whether
        # objects with the same arguments were created
        arg_key = f'{backend}'
        for key, value in kwargs.items():
            arg_key += f':{key}:{value}'

        # if a backend was overridden, it will create a new object
        if arg_key in cls._instances:
            _instance = cls._instances[arg_key]
        else:
            # create a new object and put it to _instance
            _instance = super().__new__(cls)
            if backend is not None:
                _instance.client = cls._backends[backend](**kwargs)

            cls._instances[arg_key] = _instance

        return _instance

    def __init__(
        self,
        *,
        model_path: str = None,
        base_url: str = None,
        api_key: str = None,
        gen_params: dict = None,
        backend: str = None,
        backend_config: dict = None,
    ):

        pass

    def generate(self, inputs: Union[str, List[str]], **gen_params) -> str:
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
        raise NotImplementedError

    def stream_generate(self, inputs: str, **gen_params) -> List[str]:
        """Generate results as streaming given a str inputs.

        Args:
            inputs (str):
            gen_params (dict): The input params for generation.

        Returns:
            str: A generated string.
        """
        raise NotImplementedError

    def chat(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        session_ids: Union[int, List[int]] = None,
        **gen_params,
    ):
        """Generate completion from a list of templates.

        Args:
            inputs (Union[List[dict], List[List[dict]]]):
            gen_params (dict): The input params for generation.
        Returns:
        """
        if isinstance(inputs[0], list):
            _inputs = list()
            for msg in inputs:
                _inputs.append(self.template_parser(msg))
        else:
            _inputs = self.template_parser(inputs)
        return self.generate(_inputs, **gen_params)

    def stream_chat(self, inputs: List[dict], **gen_params):
        """Generate results as streaming given a list of templates.

        Args:
            inputs (Union[List[dict]):
            gen_params (dict): The input params for generation.
        Returns:
        """
        raise NotImplementedError

    def tokenize(self, prompts: Union[str, List[str], List[dict],
                                      List[List[dict]]]):
        """Tokenize the input prompts.

        Args:
            prompts(str | List[str]): user's prompt, or a batch prompts

        Returns:
            Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray): prompt's token
            ids, ids' length and requested output length
        """
        raise NotImplementedError

    def update_gen_params(self, **kwargs):
        gen_params = copy(self.gen_params)
        gen_params.update(kwargs)
        return gen_params


class AsyncLLMMixin:

    async def generate(
        self,
        inputs: Union[str, List[str]],
        session_ids: Union[int, List[int]] = None,
        **gen_params,
    ) -> str:
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
        raise NotImplementedError

    async def stream_generate(self, inputs: str, **gen_params) -> List[str]:
        """Generate results as streaming given a str inputs.

        Args:
            inputs (str):
            gen_params (dict): The input params for generation.

        Returns:
            str: A generated string.
        """
        raise NotImplementedError

    async def chat(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        session_ids: Union[int, List[int]] = None,
        **gen_params,
    ):
        """Generate completion from a list of templates.

        Args:
            inputs (Union[List[dict], List[List[dict]]]):
            gen_params (dict): The input params for generation.
        Returns:
        """
        if isinstance(inputs[0], list):
            _inputs = list()
            for msg in inputs:
                _inputs.append(self.template_parser(msg))
        else:
            _inputs = self.template_parser(inputs)
        return await self.generate(_inputs, session_ids, **gen_params)

    async def stream_chat(self, inputs: List[dict], **gen_params):
        """Generate results as streaming given a list of templates.

        Args:
            inputs (Union[List[dict]):
            gen_params (dict): The input params for generation.
        Returns:
        """
        raise NotImplementedError

    async def tokenize(self, prompts: Union[str, List[str], List[dict],
                                            List[List[dict]]]):
        """Tokenize the input prompts.

        Args:
            prompts(str | List[str]): user's prompt, or a batch prompts

        Returns:
            Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray): prompt's token
            ids, ids' length and requested output length
        """
        raise NotImplementedError


class AsyncBaseLLM(AsyncLLMMixin, BaseLLM):
    pass


class InternLM(BaseLLM):
    base_url = 'https://internlm.com'
    model_name = 'internlm2-chat'
