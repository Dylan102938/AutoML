from eva.models.naive_bayes import NaiveBayesClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def test_naive_bayes_simple():
    config = {
        "outputs_col": "target",
        "output_mapping": {
            0: "versicolor",
            1: "setosa",
            2: "virginica"
        },
        "parameters": {
            "feature_type": "gaussian"
        }
    }

    iris = load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    X = data.drop(columns='target')
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    classifier = NaiveBayesClassifier(config)
    model = classifier.train(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nNumber of mislabeled points out of a total %d points: %d" % (X_test.shape[0], (y_test != y_pred).sum()))

    assert (y_test != y_pred).sum() < 0.15 * X_test.shape[0]

    test = pd.DataFrame(data=[[5.0, 3.3, 1.3, 0.15]], columns=data.drop(columns="target").columns)
    pred = classifier.predict(test)

    print("\nPrediction for data %s: %s" % ([5.0, 3.3, 1.3, 0.15], pred))
    assert pred == "versicolor"


def test_naive_bayes_digits():
    config = {
        "outputs_col": "target",
        "output_mapping": {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9"
        },
        "parameters": {
            "feature_type": "multinomial"
        }
    }

    digits = load_digits(return_X_y=False, as_frame=True)
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    classifier = NaiveBayesClassifier(config)
    model = classifier.train(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy score:", accuracy)

    assert accuracy > 0.85
