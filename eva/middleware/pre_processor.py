import settings
import pandas as pd
from pandas import DataFrame
from typing import Set, Any
from sklearn.preprocessing import LabelEncoder
import os


class TabularPreProcessor:
    data_table: DataFrame
    data_attrs: Set[str] = {'zeroes', 'mean', 'median', 'mode'}

    def __init__(self, table: DataFrame):
        self.data_table = table

    def to_python_list(self) -> object:
        return self.data_table.T.reset_index().values.tolist()

    def apply_func_to_col(self, col_name: str, func = lambda x: x) -> None:
        self.data_table[col_name] = self.data_table[col_name].apply(func)

    def categorize_col(self, col_name: str) -> None:
        enc = LabelEncoder()
        self.data_table[col_name] = enc.fit_transform(self.data_table[col_name])

    def replace_nan_with(self, col_name: str, data_attr: str):
        assert data_attr in self.data_attrs

        if data_attr == "zeroes":
            self.apply_func_to_col(col_name, lambda x: 0 if pd.isna(x) else x)
        elif data_attr == "mean":
            mean = self.data_table[col_name].mean(skipna=True)
            self.apply_func_to_col(col_name, lambda x: mean if pd.isna(x) else x)
        elif data_attr == "median":
            median = self.data_table[col_name].median(skipna=True)
            self.apply_func_to_col(col_name, lambda x: median if pd.isna(x) else x)
        elif data_attr == "mode":
            mode = self.data_table[col_name].mode(dropna=True)
            self.apply_func_to_col(col_name, lambda x: mode if pd.isna(x) else x)

    def to_csv(self, filename) -> None:
        self.data_table.to_csv(os.path.join(settings.HOME_DIR, filename), index=False)

    @classmethod
    def from_csv(cls, filename):
        return cls(pd.read_csv(filename))
