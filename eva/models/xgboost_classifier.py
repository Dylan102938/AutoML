from eva.models.classifier import Classifier, Config
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from typing import Any


class XGBoostClassifier(Classifier):
    default_config: Config = {
        "outputs_col": 0,
        "output_mapping": {
            0: 0,
            1: 1
        },
        "parameters": {
            'eta': 0.3,
            'gamma': 0,
            'max_depth': 6,
            'min_child_weight': 1,
            'max_delta_step': 0,
            'colsample_bytree': 1,
            'alpha': 0
        }
    }

    config: Config
    model: Any

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None

    def train(self, x: pd.DataFrame, y: pd.DataFrame) -> Any:
        params = self.config["parameters"]

        self.model = xgb.XGBClassifier(
            eta=params['eta'],
            gamma=params['gamma'],
            max_depth=int(params['max_depth']),
            min_child_weight=params['min_child_weight'],
            max_delta_step=params['max_delta_step'],
            colsample_bytree=params['colsample_bytree'],
            alpha=params['alpha']
        )

        self.model.fit(x, y)

        return self.model

    def predict(self, model_input: pd.DataFrame, custom_model: Any = None) -> Any:
        model = custom_model
        if custom_model is None:
            model = self.model

        y_pred = model.predict(model_input)

        return self.config["output_mapping"][y_pred[0]]

    def tune(self, x: pd.DataFrame, y: pd.DataFrame, debug=False) -> Any:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        gridsearch_params = {
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            # 'colsample_bytree': [0.3, 0.7]
        }

        clf = xgb.XGBClassifier()
        search = GridSearchCV(
            estimator=clf,
            param_grid=gridsearch_params,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        search.fit(x, y)

        print("Best parameters:", search.best_params_)
        self.config['parameters'].update(search.best_params_)
        return self.train(x_train, y_train)

