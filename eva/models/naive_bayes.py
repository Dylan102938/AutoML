from eva.models.classifier import Classifier, Config
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB
import pandas as pd
from typing import Any


class NaiveBayesClassifier(Classifier):
    default_config: Config = {
        "outputs_col": 0,
        "output_mapping": {
            0: 0,
            1: 1
        },
        "parameters": {
            "feature_type": "gaussian"
        }
    }

    config: Config
    model: Any

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None

    def train(self, data: pd.DataFrame) -> Any:

        models = {
            "gaussian": GaussianNB(),
            "multinomial": MultinomialNB(),
            "categorical": CategoricalNB(),
            "bernoulli": BernoulliNB()
        }

        assert self.config["parameters"]["feature_type"] in models

        x = data.drop(columns=self.config["outputs_col"])
        y = data[self.config["outputs_col"]]

        classifier = models[self.config["parameters"]["feature_type"]]
        self.model = classifier.fit(x, y)

        return self.model

    def predict(self, model_input: pd.DataFrame, custom_model: Any = None) -> Any:
        model = custom_model
        if custom_model is None:
            model = self.model

        y_pred = model.predict(model_input)

        return self.config["output_mapping"][y_pred[0]]
