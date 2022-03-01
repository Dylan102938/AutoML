import settings
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from typing import Tuple


def preprocess(df: DataFrame, y_col: str) -> Tuple[DataFrame, DataFrame]:
    x, y = df.drop(axis=1, columns=y_col), df[y_col]

    x_norm = (x - x.min()) / (x.max() - x.min())
    return x_norm, y


def pca_plt(df: DataFrame, y_col: str) -> None:
    x, y = preprocess(df, y_col)
    pca = PCA(n_components=2)
    transformed = pd.DataFrame(pca.fit_transform(x))

    for val in list(y.unique()):
        plt.scatter(transformed[y == val][0], transformed[y == val][1], label=str(val))

    plt.legend()
    plt.show()


def lda_plt(df: DataFrame, y_col: str) -> None:
    x, y = preprocess(df, y_col)

    try:
        lda = LDA(n_components=2)
        transformed = pd.DataFrame(lda.fit_transform(x, y))
    except:
        lda = LDA(n_components=1)
        transformed = pd.DataFrame(lda.fit_transform(x, y))

    for val in list(y.unique()):
        try:
            plt.scatter(transformed[y == val][0], transformed[y == val][1], label=str(val))
        except:
            plt.scatter(transformed[y == val], transformed[y == val], label=str(val))

    plt.legend()
    plt.show()


def p_coord_plt(df: DataFrame, y_col: str) -> None:
    x, y = preprocess(df, y_col)

    # Select features to include in the plot
    plot_feat = list(x.columns)

    # Concat classes with the normalized data
    data_norm = pd.concat([x[plot_feat], y], axis=1)

    # Perform parallel coordinate plot
    parallel_coordinates(data_norm, y_col)
    plt.show()
