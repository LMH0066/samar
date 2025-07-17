from typing import Literal, Tuple

import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from samar.util import (
    DEFAULT_CONFIG,
    get_clf,
    get_funcs_name,
    load_config,
    write_stable_test_result,
)


def generate_datasets(
    X: np.array,
    y: np.array,
    epoch: int = DEFAULT_CONFIG["epoch"],
    test_size: float = DEFAULT_CONFIG["test_size"],
) -> Tuple[np.array, np.array, np.array, np.array]:
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for i in range(epoch):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i + 1
        )
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    return np.array(X_trains), np.array(X_tests), np.array(y_trains), np.array(y_tests)


def train_models(
    X: np.array,
    y: np.array,
    task: Literal["classification", "regression"],
    epoch: int = DEFAULT_CONFIG["epoch"],
    funcs: dict = DEFAULT_CONFIG["funcs"],
) -> dict:
    clfs = dict()
    for func_name in get_funcs_name(funcs):
        clfs[func_name] = []
        for i in tqdm(range(epoch)):
            clf = get_clf(funcs, func_name, task, random_state=i + 1)

            clf.fit(X[i], y[i])
            clfs[func_name].append(clf)
    return clfs


def cal_accs_and_rocs(
    clfs: dict,
    X_tests: np.array,
    y_tests: np.array,
    n_class: int,
    epoch: int = DEFAULT_CONFIG["epoch"],
    funcs: dict = DEFAULT_CONFIG["funcs"],
) -> Tuple[dict, dict]:
    def cal_roc(clf, X_test: np.array, y_test: np.array, n_class: int):
        y_pred_proba = clf.predict_proba(X_test)

        if n_class == 2:
            fpr, tpr, thersholds = roc_curve(y_test, y_pred_proba[:, 1])
        else:
            # multi class
            y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
            fpr, tpr, thersholds = roc_curve(
                y_test_one_hot.ravel(), y_pred_proba.ravel()
            )
        roc_auc = auc(fpr, tpr)
        roc = [fpr, tpr, roc_auc]

        return roc

    if X_tests.ndim == 2 and y_tests.ndim == 1:
        X_tests = np.tile(X_tests, (epoch, 1, 1))
        y_tests = np.tile(y_tests, (epoch, 1))
    assert X_tests.ndim == 3 and y_tests.ndim == 2

    accs, rocs = dict(), dict()
    for func_name in get_funcs_name(funcs):
        accs[func_name], rocs[func_name] = [], []
        for i in range(epoch):
            clf, X_test, y_test = clfs[func_name][i], X_tests[i], y_tests[i]
            _acc = clf.score(X_test, y_test)
            _roc = cal_roc(clf, X_test, y_test, n_class)

            accs[func_name].append(_acc)
            rocs[func_name].append(
                {
                    "fpr": _roc[0].tolist(),
                    "tpr": _roc[1].tolist(),
                    "auc": _roc[2],
                }
            )

    return accs, rocs


def cal_accs_and_pccs(
    clfs: dict,
    X_tests: np.array,
    y_tests: np.array,
    epoch: int = DEFAULT_CONFIG["epoch"],
    funcs: dict = DEFAULT_CONFIG["funcs"],
) -> Tuple[dict, dict]:
    def cal_pcc(clf, X_test: np.array, y_test: np.array):
        y_pred = clf.predict(X_test)
        correlation, p_value = pearsonr(y_pred, y_test)

        return correlation, p_value

    if X_tests.ndim == 2 and y_tests.ndim == 1:
        X_tests = np.tile(X_tests, (epoch, 1, 1))
        y_tests = np.tile(y_tests, (epoch, 1))
    assert X_tests.ndim == 3 and y_tests.ndim == 2

    accs, pccs = dict(), dict()
    for func_name in get_funcs_name(funcs):
        accs[func_name], pccs[func_name] = [], []
        for i in range(epoch):
            clf, X_test, y_test = clfs[func_name][i], X_tests[i], y_tests[i]
            _acc = clf.score(X_test, y_test)
            _pcc = cal_pcc(clf, X_test, y_test)

            accs[func_name].append(_acc)
            pccs[func_name].append({"r": _pcc[0], "p": _pcc[1]})

    return accs, pccs


def stable_test(
    X: np.array,
    y: np.array,
    task: Literal["classification", "regression"],
    output_path: str = None,
    config_path: str = None,
) -> Tuple[dict, dict]:
    config = load_config(config_path) if config_path else DEFAULT_CONFIG

    X_trains, X_tests, y_trains, y_tests = generate_datasets(
        X, y, config["epoch"], config["test_size"]
    )
    clfs = train_models(X_trains, y_trains, task, config["epoch"], config["funcs"])

    if task == "classification":
        accs, rocs = cal_accs_and_rocs(
            clfs,
            X_tests,
            y_tests,
            n_class=np.unique(y).size,
            epoch=config["epoch"],
            funcs=config["funcs"],
        )
        scores = {"accs": accs, "rocs": rocs}
    elif task == "regression":
        accs, pccs = cal_accs_and_pccs(
            clfs,
            X_tests,
            y_tests,
            epoch=config["epoch"],
            funcs=config["funcs"],
        )
        scores = {"accs": accs, "pccs": pccs}

    if output_path:
        write_stable_test_result(output_path, scores)
    return scores, clfs


def predict(clfs: dict, X_tests: np.array) -> dict:
    y_preds = dict()
    for func_name in clfs.keys():
        _y_preds = (
            pd.DataFrame([clf.predict(X_tests) for clf in clfs[func_name]])
            .mode()
            .iloc[0]
        )
        y_preds[func_name] = _y_preds.values
    return y_preds


def cal_shap(
    X: np.array,
    y: np.array,
    task: Literal["classification", "regression"],
    config_path: str = None,
) -> dict:
    config = load_config(config_path) if config_path else DEFAULT_CONFIG

    X_trains, X_tests, y_trains, y_tests = generate_datasets(
        X, y, config["epoch"], config["test_size"]
    )
    clfs = train_models(X_trains, y_trains, task, config["epoch"], config["funcs"])

    shap_values = dict()
    for func in clfs.keys():
        for i in range(len(clfs[func])):
            clf = clfs[func][i]
            X_train, X_test = X_trains[i], X_tests[i]
            X_train_summary = shap.kmeans(X_train, 10)
            ex = shap.KernelExplainer(clf.predict, X_train_summary)
            _shap_values = ex.shap_values(X_test)

            shap_values[func] = (
                np.concatenate((shap_values[func], _shap_values), axis=0)
                if func in shap_values
                else _shap_values
            )

    return shap_values
