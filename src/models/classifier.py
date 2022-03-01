import settings
import os
import json
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from typing_extensions import TypedDict


class Config(TypedDict):
    data_path: str
    batch_size: int
    epochs: int
    outputs: List[str]
    output_mapping: Dict[int, Any]
    split_ratio: float


class Classifier(ABC):

    configs: Config

    def __init__(self, config: Config):
        self.configs = config

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def predict(self, model_input: Any) -> Any:
        ...

    @abstractmethod
    def test(self) -> None:
        ...

    @classmethod
    def from_json(cls, filepath: str):
        with open(os.path.join(settings.HOME_DIR, filepath)) as f:
            return cls(json.load(f))
