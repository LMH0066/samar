import os

import pandas as pd
import pytest

from samar.compute import cal_RF_feature_importance, stable_test
from samar.util import load_xlsx, read_stable_test_result


@pytest.mark.parametrize(
    "xlsx_path, preprocess_func, expected_file_path",
    [
        (
            "tests/file/SSNHL.xlsx",
            "default",
            "tests/file/stable_test_result_default.npy",
        ),
        ("tests/file/SSNHL.xlsx", "KNN", "tests/file/stable_test_result_KNN.npy"),
    ],
)
def test_stable_test(dir, xlsx_path, preprocess_func, expected_file_path):
    X, y, _ = load_xlsx(xlsx_path, preprocess_func)
    accs, rocs = stable_test(
        X,
        y,
        output_path=os.path.join(
            dir, "stable_test_result_{}.npy".format(preprocess_func)
        ),
    )
    truth_accs, truth_rocs = read_stable_test_result(expected_file_path)

    assert accs == truth_accs
    assert rocs == truth_rocs


@pytest.mark.parametrize(
    "xlsx_path, preprocess_func, expected_file_path",
    [
        (
            "tests/file/SSNHL.xlsx",
            "default",
            "tests/file/RF_feature_importance_result_default.csv",
        ),
        (
            "tests/file/SSNHL.xlsx",
            "KNN",
            "tests/file/RF_feature_importance_result_KNN.csv",
        ),
    ],
)
def test_cal_RF_feature_importance(dir, xlsx_path, preprocess_func, expected_file_path):
    X, y, filter_data = load_xlsx(xlsx_path, preprocess_func)

    importance = cal_RF_feature_importance(
        X,
        y,
        filter_data.columns,
        output_path=os.path.join(
            dir, "RF_feature_importance_result_{}.csv".format(preprocess_func)
        ),
    )
    truth_importance = pd.read_csv(expected_file_path, index_col=0)

    pd.testing.assert_frame_equal(importance, truth_importance)
