from typing import Dict, List

from lagent.memory import Memory
from lagent.registry import AGGREGATOR_REGISTRY, AutoRegister


class DefaultAggregator(metaclass=AutoRegister(AGGREGATOR_REGISTRY)):

    def aggregate(self,
                  messages: Memory,
                  name: str,
                  system_instruction: str = None) -> List[Dict[str, str]]:
        _message = []
        messages = messages.get_memory()
        if system_instruction:
            _message.append(
                dict(role='system', content=str(system_instruction)))
        for message in messages:
            if message.sender == name:
                _message.append(
                    dict(role='assistant', content=str(message.content)))
            else:
                user_message = message.content
                if len(_message) > 0 and _message[-1]['role'] == 'user':
                    _message[-1]['content'] += user_message
                else:
                    _message.append(dict(role='user', content=user_message))
        return _message
