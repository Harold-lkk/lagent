import argparse
import importlib
import json
import logging
import sys
from typing import List, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from lagent.schema import AgentMessage


def load_class_from_string(class_path: str, path=None):
    path_in_sys = False
    if path:
        if path not in sys.path:
            path_in_sys = True
            sys.path.insert(0, path)  # Temporarily add the path to sys.path

    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls
    finally:
        if path and path_in_sys:
            sys.path.remove(
                path)  # Ensure to clean up by removing the path from sys.path


class AsyncAgentAPIServer:

    def __init__(self, config: dict, host: str = '0.0.0.0', port: int = 8090):
        self.app = FastAPI(docs_url='/')
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )
        cls_name = config.pop('type')
        python_path = config.pop('python_path', None)
        cls_name = load_class_from_string(cls_name, python_path) if isinstance(
            cls_name, str) else cls_name
        self.agent = cls_name(**config)
        self.setup_routes()
        self.run(host, port)

    def setup_routes(self):

        @self.app.post('/chat_completion')
        async def process_message(*message: Union[AgentMessage,
                                                  List[AgentMessage]],
                                  session_id: str = Query('0')):
            try:
                # Ensure the agent call is correctly awaited and results are returned properly.
                result = await self.agent(*message, session_id=session_id)
                return result
            except Exception as e:
                logging.error(f'Error processing message: {str(e)}')
                raise HTTPException(
                    status_code=500, detail='Internal Server Error')

    def run(self, host='0.0.0.0', port=8090):
        logging.info(f'Starting server at {host}:{port}')
        uvicorn.run(self.app, host=host, port=port)


def parse_args():
    parser = argparse.ArgumentParser(description='Async Agent API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument(
        '--config',
        type=json.loads,
        required=True,
        help='JSON configuration for the agent')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    AsyncAgentAPIServer(args.config, host=args.host, port=args.port)