import importlib

import numpy as np
import pandas as pd
import yaml
from fancyimpute import KNN

"""
config.yaml related code
"""


def load_config(path: str) -> dict:
    return yaml.safe_load(open(path, "r"))


def get_funcs_name(funcs: dict):
    return funcs.keys()


def get_clf(funcs, func_name, random_state):
    def _import(class_path: str):
        module_name, class_name = class_path.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), class_name)

    func = funcs[func_name]
    return _import(func["class_path"])(**func["kwargs"], random_state=random_state)


"""
compute related code
"""


def load_xlsx(
    path: str, preprocess_func: str, result_col: str = "efficacy evaluation"
) -> (np.array, np.array, pd.DataFrame):
    data = pd.read_excel(path, index_col=0, header=[0])

    data = data.dropna(subset=[result_col])
    y = data[result_col].copy()
    data = data.drop(["efficacy evaluation"], axis=1)

    filter_data = Preprocess().do(data, preprocess_func)
    X = np.array(filter_data)
    y = np.array(y[filter_data.index])
    return X, y, filter_data


class Preprocess:
    def __init__(self):
        self.function_map = {
            "default": self.default,
            "KNN": self.KNN,
        }

    def do(self, data: pd.DataFrame, func_name: str) -> pd.DataFrame:
        filter_data = self.function_map[func_name](data)

        for column in filter_data.columns:
            filter_data[column] = pd.to_numeric(filter_data[column])

        return filter_data

    def _clean_row(self, data: pd.DataFrame, string: str) -> pd.DataFrame:
        return data[
            ~data.apply(lambda row: row.astype(str).str.contains(string).any(), axis=1)
        ]

    def default(self, data: pd.DataFrame) -> pd.DataFrame:
        filter_data = data.copy()

        filter_data = filter_data.dropna()
        return filter_data

    def KNN(self, data: pd.DataFrame) -> pd.DataFrame:
        filter_data = data.copy()

        filled_data = KNN(k=10).fit_transform(filter_data)
        filter_data[:] = filled_data

        return filter_data


def write_stable_test_result(path: str, accs: dict, rocs: dict):
    if not path.endswith(".npy"):
        path += ".npy"
    np.save(path, {"accs": accs, "rocs": rocs})


def read_stable_test_result(path: str) -> (dict, dict):
    result = np.load(path, allow_pickle=True).item()
    return result["accs"], result["rocs"]
