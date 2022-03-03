from eva.models.xgboost_classifier import XGBoostClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


def test_xgboost_simple():
    config = {
        "outputs_col": "PRICE",
        "output_mapping": {
            0: "versicolor",
            1: "setosa",
            2: "virginica"
        }
    }

    digits = load_digits(return_X_y=False, as_frame=True)
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    classifier = XGBoostClassifier(config)
    model = classifier.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Pre-tuned model accuracy:", accuracy)

    model = classifier.tune(X, y)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("After hyperparameter tuning accuracy:", accuracy)
