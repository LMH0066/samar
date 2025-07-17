import click

from tests.test_analyse import test_get_comprehensive_comparison, test_rocsplot
from tests.test_compute import test_cal_shap, test_stable_test_and_predict


@click.group()
def cli():
    pass


@cli.command()
def main():
    test_stable_test_and_predict(
        "tests/file",
        "tests/file/SSNHL.xlsx",
        "tests/config.yaml",
        "default",
        "classification",
        "tests/file/stable_test_result_default.npy",
    )
    test_stable_test_and_predict(
        "tests/file",
        "tests/file/SSNHL.xlsx",
        "tests/config.yaml",
        "KNN",
        "classification",
        "tests/file/stable_test_result_KNN.npy",
    )
    test_stable_test_and_predict(
        "tests/file",
        "tests/file/SSNHL.xlsx",
        "tests/config.yaml",
        "miNNseq",
        "classification",
        "tests/file/stable_test_result_miNNseq.npy",
    )
    test_cal_shap(
        "tests/file",
        "tests/file/SSNHL.xlsx",
        "tests/config.yaml",
        "default",
        "classification",
        "tests/file/shap_default.npy",
    )
    test_cal_shap(
        "tests/file",
        "tests/file/SSNHL.xlsx",
        "tests/config.yaml",
        "KNN",
        "classification",
        "tests/file/shap_KNN.npy",
    )
    test_rocsplot(
        "tests/file",
        "tests/file/stable_test_result_default.npy",
    )
    test_rocsplot(
        "tests/file",
        "tests/file/stable_test_result_KNN.npy",
    )
    test_get_comprehensive_comparison(
        "tests/file",
        "tests/file/stable_test_result_default.npy",
        "tests/file/comprehensive_result_default.csv",
    )
    test_get_comprehensive_comparison(
        "tests/file",
        "tests/file/stable_test_result_KNN.npy",
        "tests/file/comprehensive_result_KNN.csv",
    )


if __name__ == "__main__":
    cli()
