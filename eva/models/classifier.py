import settings
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Union, Dict
from typing_extensions import TypedDict


class Config(TypedDict):
    outputs_col: Union[str, int]
    output_mapping: Dict[int, Any]
    parameters: Dict[str, Any]


class Classifier(ABC):
    default_config: Any = {}
    config: Any

    def __init__(self, config: Any):
        self.config = dict(self.default_config)
        self.config.update(config)

    @abstractmethod
    def train(self, x: Any, y: Any) -> Any:
        ...

    @abstractmethod
    def predict(self, model_input: Any) -> Any:
        ...

    @classmethod
    def from_json(cls, filepath: str):
        with open(os.path.join(settings.HOME_DIR, filepath)) as f:
            return cls(json.load(f))
