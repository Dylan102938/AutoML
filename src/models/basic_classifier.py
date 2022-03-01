import settings
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from typing import Tuple, Any, List
import json
import pandas as pd
import os

from src.models.classifier import Classifier, Config


class BasicClassifier(Classifier):
    """
    Input: CSV path
    Output: Classification on one column
    """

    data_path: str
    features_df: DataFrame
    outputs_df: DataFrame
    training_set: Tuple[Any, Any]
    testing_set: Tuple[Any, Any]
    classifier: Sequential

    def __init__(self, config: Config):
        super().__init__(config)

        self.data_path = os.path.join(settings.HOME_DIR, self.configs["data_path"])

        data = pd.read_csv(self.data_path)
        self.classifier = Sequential()
        self.outputs_df = DataFrame(data[self.configs["outputs"][0]])
        self.features_df = data.drop(columns=self.configs["outputs"][0])

        x_train, x_test, y_train, y_test = train_test_split(
            self.features_df,
            self.outputs_df,
            test_size=self.configs["split_ratio"]
        )

        self.training_set = (x_train, y_train)
        self.testing_set = (x_test, y_test)

    def train(self) -> None:
        x_train, y_train = self.training_set
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)

        num_nodes = (len(self.features_df.columns) + len(self.outputs_df.columns)) // 2
        self.classifier.add(
            Dense(
                num_nodes,
                kernel_initializer="uniform",
                activation="relu",
                input_dim=len(self.features_df.columns)
            )
        )
        self.classifier.add(
            Dense(
                len(self.outputs_df.columns),
                kernel_initializer="uniform",
                activation="sigmoid"
            )
        )

        self.classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.classifier.fit(x_train, y_train, batch_size=self.configs["batch_size"], epochs=self.configs["epochs"])

    def predict(self, model_input: List[Any]) -> Any:
        return 0

    def test(self) -> None:
        x_test, y_test = self.testing_set
        y_pred = self.classifier.predict(x_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
