import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from samar.util import get_clf, get_funcs_name, load_config, write_stable_test_result


def generate_datasets(X, y, epoch, test_size):
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for i in range(epoch):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i + 1
        )
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    return X_trains, X_tests, y_trains, y_tests


def train_models(X, y, epoch, funcs):
    clfs = dict()
    for func_name in get_funcs_name(funcs):
        clfs[func_name] = []
        for i in range(epoch):
            clf = get_clf(funcs, func_name, random_state=i + 1)

            clf.fit(X[i], y[i])
            clfs[func_name].append(clf)
    return clfs


def cal_acc_and_roc(clf, X_test, y_test, n_class: int):
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


def cal_accs_and_rocs(X, y, epoch, test_size, funcs) -> (dict, dict):
    X_trains, X_tests, y_trains, y_tests = generate_datasets(X, y, epoch, test_size)
    clfs = train_models(X_trains, y_trains, epoch, funcs)

    accs, rocs = dict(), dict()
    for func_name in get_funcs_name(funcs):
        accs[func_name], rocs[func_name] = [], []
        for i in range(epoch):
            clf, X_test, y_test = clfs[func_name][i], X_tests[i], y_tests[i]
            _acc, _roc = cal_acc_and_roc(clf, X_test, y_test, np.unique(y).size)

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
) -> (dict, dict):
    config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    accs, rocs = cal_accs_and_rocs(X, y, **config)

    if output_path:
        write_stable_test_result(output_path, accs, rocs)
    return accs, rocs


def cal_RF_feature_importance(
    X: np.array,
    y: np.array,
    columns_name: list,
    output_path: str = None,
    config_path: str = None,
) -> pd.DataFrame:
    config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

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
