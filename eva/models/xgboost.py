from eva.models.classifier import Classifier, Config
import xgboost as xgb
import pandas as pd
from typing import Any
from hyperopt import Trials, fmin, hp, tpe


class XGBoostClassifier(Classifier):
    default_config: Config = {
        "outputs_col": 0,
        "output_mapping": {
            0: 0,
            1: 1
        },
        "parameters": {
            "learning_rate": 0.1,
            "max_depth": 5,
            "colsample_bytree": 0.3,
            "n_estimators": 10,
            "objective": "reg:logistic",
            "alpha": 10
        }
    }

    config: Config
    model: Any

    def __init__(self, config: Config):
        super().__init__(config)
        self.model = None

    def train(self, data: pd.DataFrame) -> Any:
        x = data.drop(columns=self.config["outputs_col"])
        y = data[self.config["outputs_col"]]
        params = self.config["parameters"]
        data_dmatrix = xgb.DMatrix(data=x, label=y)

        self.model = xgb.XGBClassifier(
            objective=params["objective"],
            colsample_bytree=params["colsample_bytree"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            alpha=params["alpha"],
            n_estimators=params["n_estimators"]
        )

        self.model.fit(x, y)

        return self.model

        # 3-fold cross validation

    def predict(self, model_input: pd.DataFrame, custom_model: Any = None) -> Any:
        model = custom_model
        if custom_model is None:
            model = self.model

        y_pred = model.predict(model_input)

        return self.config["output_mapping"][y_pred[0]]

    def tune(self, data: pd.DataFrame) -> Any:
        x = data.drop(columns=self.config["outputs_col"])
        y = data[self.config["outputs_col"]]
        space = {
            'max_depth': hp.quniform("max_depth", 3, 18, 1),
            'gamma': hp.uniform('gamma', 1, 9),
            'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
            'n_estimators': 180,
            'seed': 0
        }

        trials = Trials()
        best_hyperparams = fmin(
            fn=self.config["parameters"]["objective"],
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials
        )

        self.config["parameters"].update(best_hyperparams)
        self.train()

        data_dmatrix = xgb.DMatrix(data=x, label=y)

