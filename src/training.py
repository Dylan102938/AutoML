import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import middleware

SPLIT_RATIO = 0.3
KERNEL_INITIALIZER = "uniform"
BATCH_SIZE = 10
EPOCHS = 1


def train_classifier(x, y, cols_x, cols_y, out_results=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=SPLIT_RATIO)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    classifier = Sequential()

    num_nodes = (cols_x + cols_y) // 2

    classifier.add(
        Dense(
            num_nodes,
            kernel_initializer=KERNEL_INITIALIZER,
            activation="relu",
            input_dim=cols_x
        )
    )
    classifier.add(
        Dense(
            cols_y,
            kernel_initializer=KERNEL_INITIALIZER,
            activation="sigmoid"
        )
    )

    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    if out_results:
        y_pred = classifier.predict(x_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    return classifier


if __name__ == "__main__":
    with open("../tests/configs/input_structure/input-3.json") as f:
        data_format = f.read()

    df = middleware.json_to_pandas(data_format)
    cols = len(df.columns)
    X = df.drop(['Class'], axis=1).values
    Y = df['Class'].values

    classifier = train_classifier(X, Y, cols-1, 1)
