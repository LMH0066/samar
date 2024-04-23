import importlib
import inspect
import os
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from fancyimpute import KNN

"""
config.yaml related code
"""


def load_config(path: str) -> dict:
    return yaml.safe_load(open(path, "r"))


DEFAULT_CONFIG = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))


def get_funcs_name(funcs: dict):
    return funcs.keys()


def get_clf(funcs: dict, func_name: str, task: str, random_state: int):
    def _import(class_path: str):
        module_name, class_name = class_path.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), class_name)

    def _filter_args(obj, args):
        accepted_args = inspect.signature(obj).parameters
        filtered_args = {
            key: value for key, value in args.items() if key in accepted_args
        }
        return filtered_args

    func = funcs[func_name].copy()
    func["kwargs"].update(dict(random_state=random_state))

    obj = _import(func["class_path"][task])
    args = _filter_args(obj, func["kwargs"])
    return obj(**args)


"""
compute related code
"""


def load_xlsx(
    path: str, preprocess_func: str, result_col: str = "efficacy evaluation"
) -> Tuple[np.array, np.array, pd.DataFrame]:
    data = pd.read_excel(path, index_col=0, header=[0])

    data = data.dropna(subset=[result_col])
    y = data[result_col].copy()
    data = data.drop([result_col], axis=1)

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


def write_stable_test_result(path: str, scores: dict):
    if not path.endswith(".npy"):
        path += ".npy"
    np.save(path, scores)


def read_stable_test_result(path: str) -> dict:
    scores = np.load(path, allow_pickle=True).item()
    return scores
