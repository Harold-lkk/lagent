

from copy import deepcopy
from typing import Callable, List, Union, Dict
from lagent.llms.schema import ChatCompletionResponse

class LocalBackend:
    
    def __init__(self,
                 model,
                 return_tyep: str ='str',
                 gen_params: dict = None,
                 backend_config: dict = None,) -> None:

        self.model = model
        self.gen_params = gen_params
        self.backend_config = backend_config
        self.return_type = return_tyep
        self.client = self.init_client(model, gen_params, backend_config)

    def init_client(self, model, gen_params, backend_config) -> Callable:
        raise NotImplementedError()

    def completion(
        self,
        inputs: Union[str, List[str]],
        **gen_params,
    ) -> Union[str, List[str], ]:
        raise NotImplementedError()

    def chat_completion(
            self,
            inputs: Union[List[dict], List[List[dict]]],
            **gen_params,
        ) -> Union[str, List[str], ChatCompletionResponse, List[ChatCompletionResponse]]:
        raise NotImplementedError()

    def generate_params_map(self, gen_params):
        raise NotImplementedError()

    def update_gen_params(self, gen_params):
        gen_params = deepcopy(self.gen_params)
        gen_params.update(gen_params)
        gen_params = self.generate_params_map(gen_params)
        return gen_params

    def whether_support_model(self, model_name):
        raise NotImplementedError()

    def response_to_chat_completion_response(self, response):
        raise NotImplementedError()
    
    def response_to_completion_responses(self, response):
        raise NotImplementedError()

class RemoteBackend(LocalBackend):

    def __init__(self,
                 model: str,
                 base_url: str,
                 api_key: str,
                 proxy: str,
                 gen_params: Dict = None,
                 backend_config: Dict = None,) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.backend_config = backend_config
        self.gen_params = gen_params
        self.proxy = proxy


class AsyncMixin:

    async def completion(
        self,
        inputs: Union[str, List[str]],
        **gen_params,
    ) -> Union[str, List[str]]:
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

    async def chat_completion(
        self,
        inputs: Union[List[dict], List[List[dict]]],
        **gen_params,
    ) -> Union[str, List[str]]:
        """Generate completion from a list of templates.

        Args:
            inputs (Union[List[dict], List[List[dict]]]):
            gen_params (dict): The input params for generation.
        Returns:
        """
        raise NotImplementedError
