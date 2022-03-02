import settings
import pandas as pd
from typing import List, Any
import os

from eva.models.classifier import Classifier, Config


class CNNClassifier(Classifier):
    """
    Input: image folder with folder name as classification
    Output: image -> classification
    """

    data_path: str

    def __init__(self, config: Config):
        super().__init__(config)

        self.data_path = os.path.join(settings.HOME_DIR, self.configs["data_path"])

    def train(self) -> None:
        print("Hello World")
        ln = LogScaler()


    def predict(self, model_input: List[Any]) -> Any:
        return 0

    def test(self) -> None:
        print("Hello World")
