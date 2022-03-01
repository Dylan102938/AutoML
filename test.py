import settings
import os
import pandas as pd
from src.middleware.pre_processor import TabularPreProcessor
from src.models.basic_classifier import BasicClassifier
import numpy as np

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    pp = TabularPreProcessor.from_csv(os.path.join(settings.HOME_DIR, "tests/data/titanic/train.csv"))
    print("Checking to see if all columns are nan-free...")
    for col in list(pp.data_table.columns):
        print(col + ":", not pp.data_table[col].isnull().values.any())

    print("\nProcessing...")

    pp.data_table = pp.data_table[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

    # pp.replace_nan_with("Cabin", "zeroes")
    pp.replace_nan_with("Age", "mean")
    pp.data_table = pp.data_table.dropna()

    pp.categorize_col("Sex")

    # def alphanumeric(x):
    #     if x == 0:
    #         return x
    #     else:
    #         try:
    #             return (ord(x[0]) - ord('A'))*1000 + int(x[1:])
    #         except:
    #             return -1
    #
    # pp.apply_func_to_col("Cabin", alphanumeric)
    # pp.data_table.drop(pp.data_table[pp.data_table['Cabin'] == -1].index, inplace=True)
    # print("")

    print("Checking to see if all columns are nan-free...")
    for col in list(pp.data_table.columns):
        print(col + ":", not pp.data_table[col].isnull().values.any())

    pp.to_csv(os.path.join(settings.HOME_DIR, "tests/data/titanic/train-filtered.csv"))

    b = BasicClassifier.from_json(os.path.join("tests/configs/model_configs/model-config-1.json"))
    b.train()
    b.test()
