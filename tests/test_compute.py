import os

import pandas as pd
import pytest

from samar.compute import cal_RF_feature_importance, predict, stable_test
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
    "xlsx_path, preprocess_func, task, expected_file_path",
    [
        (
            "tests/file/SSNHL.xlsx",
            "default",
            "classification",
            "tests/file/RF_feature_importance_result_default.csv",
        ),
        (
            "tests/file/SSNHL.xlsx",
            "KNN",
            "classification",
            "tests/file/RF_feature_importance_result_KNN.csv",
        ),
    ],
)
def test_cal_RF_feature_importance(
    dir, xlsx_path, preprocess_func, task, expected_file_path
):
    X, y, filter_data = load_xlsx(xlsx_path, preprocess_func)

    importance = cal_RF_feature_importance(
        X,
        y,
        filter_data.columns,
        task,
        output_path=os.path.join(
            dir, "RF_feature_importance_result_{}.csv".format(preprocess_func)
        ),
    )
    truth_importance = pd.read_csv(expected_file_path, index_col=0)

    pd.testing.assert_frame_equal(importance, truth_importance)
