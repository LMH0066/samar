import importlib
import inspect
import os
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from samar.miNNseq import miNNseq

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

    filter_data = Preprocess().do(data, y, preprocess_func)
    X = np.array(filter_data)
    y = np.array(y[filter_data.index])
    return X, y, filter_data


class Preprocess:
    def __init__(self):
        self.function_map = {
            "default": self.default,
            "KNN": self.KNN,
            "miNNseq": self.miNNseq,
        }

    def do(self, data: pd.DataFrame, y: pd.Series, func_name: str) -> pd.DataFrame:
        filter_data = self.function_map[func_name](data, y)

        for column in filter_data.columns:
            filter_data[column] = pd.to_numeric(filter_data[column])

        return filter_data

    def _clean_row(self, data: pd.DataFrame, string: str) -> pd.DataFrame:
        return data[
            ~data.apply(lambda row: row.astype(str).str.contains(string).any(), axis=1)
        ]

    def _find_k(self, data: pd.DataFrame, y: pd.Series) -> int:
        data = data.copy()
        data = data.dropna()
        y = y[data.index].copy()

        grid = GridSearchCV(
            KNeighborsClassifier(),
            {"n_neighbors": np.arange(1, int(np.sqrt(data.shape[0])) + 1)},
            cv=5,
        )
        grid.fit(np.array(data), np.array(y))

        return grid.best_params_["n_neighbors"]

    def default(self, data: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        filter_data = data.copy()

        filter_data = filter_data.dropna()
        return filter_data

    def KNN(self, data: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        filter_data = data.copy()

        n_neighbors = self._find_k(filter_data, y)
        filled_data = KNNImputer(n_neighbors=n_neighbors).fit_transform(filter_data)
        filter_data[:] = filled_data
        return filter_data

    def miNNseq(self, data: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        filter_data = data.copy()

        n_neighbors = self._find_k(filter_data, y)
        filled_data = miNNseq(data.values, n_neighbors)
        filter_data[:] = filled_data
        return filter_data


def write_stable_test_result(path: str, scores: dict):
    if not path.endswith(".npy"):
        path += ".npy"
    np.save(path, scores)


def read_stable_test_result(path: str) -> dict:
    scores = np.load(path, allow_pickle=True).item()
    return scores
