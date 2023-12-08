import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

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
) -> (np.array, np.array, np.array, np.array):
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
    epoch: int = DEFAULT_CONFIG["epoch"],
    funcs: dict = DEFAULT_CONFIG["funcs"],
) -> dict:
    clfs = dict()
    for func_name in get_funcs_name(funcs):
        clfs[func_name] = []
        for i in range(epoch):
            clf = get_clf(funcs, func_name, random_state=i + 1)

            clf.fit(X[i], y[i])
            clfs[func_name].append(clf)
    return clfs


def cal_acc_and_roc(clf, X_test: np.array, y_test: np.array, n_class: int):
    acc = clf.score(X_test, y_test)
    y_pred_proba = clf.predict_proba(X_test)

    if n_class == 2:
        fpr, tpr, thersholds = roc_curve(y_test, y_pred_proba[:, 1])
    else:
        # multi class
        y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
        fpr, tpr, thersholds = roc_curve(y_test_one_hot.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    roc = [fpr, tpr, roc_auc]

    return acc, roc


def cal_accs_and_rocs(
    clfs: dict,
    X_tests: np.array,
    y_tests: np.array,
    n_class: int,
    epoch: int = DEFAULT_CONFIG["epoch"],
    funcs: dict = DEFAULT_CONFIG["funcs"],
) -> (dict, dict):
    accs, rocs = dict(), dict()
    for func_name in get_funcs_name(funcs):
        accs[func_name], rocs[func_name] = [], []
        for i in range(epoch):
            clf, X_test, y_test = clfs[func_name][i], X_tests[i], y_tests[i]
            _acc, _roc = cal_acc_and_roc(clf, X_test, y_test, n_class)

            accs[func_name].append(_acc)
            rocs[func_name].append(
                {
                    "fpr": _roc[0].tolist(),
                    "tpr": _roc[1].tolist(),
                    "auc": _roc[2],
                }
            )

    return accs, rocs


def stable_test(
    X: np.array,
    y: np.array,
    output_path: str = None,
    config_path: str = None,
) -> (dict, dict, dict):
    config = load_config(config_path) if config_path else DEFAULT_CONFIG

    X_trains, X_tests, y_trains, y_tests = generate_datasets(
        X, y, config["epoch"], config["test_size"]
    )
    clfs = train_models(X_trains, y_trains, config["epoch"], config["funcs"])

    accs, rocs = cal_accs_and_rocs(
        clfs,
        X_tests,
        y_tests,
        n_class=np.unique(y).size,
        epoch=config["epoch"],
        funcs=config["funcs"],
    )

    if output_path:
        write_stable_test_result(output_path, accs, rocs)
    return accs, rocs, clfs


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


def cal_RF_feature_importance(
    X: np.array,
    y: np.array,
    columns_name: list,
    output_path: str = None,
    config_path: str = None,
) -> pd.DataFrame:
    config = load_config(config_path) if config_path else DEFAULT_CONFIG

    X_trains, _, y_trains, _ = generate_datasets(
        X, y, config["epoch"], config["test_size"]
    )
    clfs = train_models(
        X_trains,
        y_trains,
        config["epoch"],
        {"RandomForest": config["funcs"]["RandomForest"]},
    )["RandomForest"]

    results = pd.DataFrame(
        [clf.feature_importances_ for clf in clfs], columns=columns_name
    )

    if output_path:
        results.to_csv(output_path)
    return results
