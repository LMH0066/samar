import os

import pandas as pd
import pytest

from samar.analyse import get_comprehensive_comparison, rocsplot
from samar.util import read_stable_test_result


@pytest.mark.parametrize(
    "stable_test_result_path",
    [
        "tests/file/stable_test_result_default.npy",
        "tests/file/stable_test_result_KNN.npy",
    ],
)
def test_rocsplot(dir, stable_test_result_path):
    scores = read_stable_test_result(stable_test_result_path)

    rocsplot(scores["rocs"], os.path.join(dir, "ROC.pdf"), False)


@pytest.mark.parametrize(
    "stable_test_result_path, expected_file_path",
    [
        (
            "tests/file/stable_test_result_default.npy",
            "tests/file/comprehensive_result_default.csv",
        ),
        (
            "tests/file/stable_test_result_KNN.npy",
            "tests/file/comprehensive_result_KNN.csv",
        ),
    ],
)
def test_get_comprehensive_comparison(dir, stable_test_result_path, expected_file_path):
    scores = read_stable_test_result(stable_test_result_path)

    comprehensive_result = get_comprehensive_comparison(
        scores,
        os.path.join(dir, os.path.basename(expected_file_path)),
    )

    truth_comprehensive_result = pd.read_csv(expected_file_path, index_col=0)
    pd.testing.assert_frame_equal(comprehensive_result, truth_comprehensive_result)
