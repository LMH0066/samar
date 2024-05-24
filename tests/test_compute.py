import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from samar.compute import cal_shap, predict, stable_test
from samar.util import load_xlsx, read_stable_test_result


@pytest.mark.parametrize(
    "xlsx_path, preprocess_func, task, expected_file_path",
    [
        (
            "tests/file/SSNHL.xlsx",
            "default",
            "classification",
            "tests/file/stable_test_result_default.npy",
        ),
        (
            "tests/file/SSNHL.xlsx",
            "KNN",
            "classification",
            "tests/file/stable_test_result_KNN.npy",
        ),
    ],
)
def test_stable_test_and_predict(
    dir, xlsx_path, preprocess_func, task, expected_file_path
):
    X, y, _ = load_xlsx(xlsx_path, preprocess_func)
    scores, clfs = stable_test(
        X,
        y,
        task,
        output_path=os.path.join(
            dir, "stable_test_result_{}.npy".format(preprocess_func)
        ),
    )
    truth_scores = read_stable_test_result(expected_file_path)

    # There are subtle differences in the results of sklearn operations on different systems
    def _is_similar(df1: pd.DataFrame, df2: pd.DataFrame):
        return (df1.mean() - df2.mean()).abs().le(0.01).all().all()

    assert _is_similar(pd.DataFrame(scores["accs"]), pd.DataFrame(truth_scores["accs"]))
    assert _is_similar(
        pd.DataFrame(scores["rocs"]).map(lambda x: x["auc"]),
        pd.DataFrame(truth_scores["rocs"]).map(lambda x: x["auc"]),
    )

    predict(clfs, X)


@pytest.mark.parametrize(
    "xlsx_path, config_path, preprocess_func, task, expected_file_path",
    [
        (
            "tests/file/SSNHL.xlsx",
            "tests/config.yaml",
            "default",
            "classification",
            "tests/file/shap_default.npz",
        ),
        (
            "tests/file/SSNHL.xlsx",
            "tests/config.yaml",
            "KNN",
            "classification",
            "tests/file/shap_KNN.npz",
        ),
    ],
)
def test_cal_shap(
    dir, xlsx_path, config_path, preprocess_func, task, expected_file_path
):
    X, y, _ = load_xlsx(xlsx_path, preprocess_func)

    importance = cal_shap(X, y, task, config_path)
    truth_importance = np.load(expected_file_path)

    def _is_similar(array1: np.array, array2: np.array):
        return np.diag(cosine_similarity(array1, array2)).mean() >= 0.95

    for key in importance:
        assert _is_similar(importance[key], truth_importance[key])
