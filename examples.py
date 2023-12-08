import click
import numpy as np

from samar.analyse import get_comprehensive_comparison, rocsplot
from samar.compute import (
    cal_accs_and_rocs,
    cal_RF_feature_importance,
    generate_datasets,
    predict,
    stable_test,
    train_models,
)
from samar.util import load_xlsx


@click.group()
def cli():
    pass


@cli.command()
@click.option("--xlsx_path")
@click.option("--preprocess_func", default="default")
def general_process(xlsx_path, preprocess_func):
    X, y, _ = load_xlsx(xlsx_path, preprocess_func)

    # stable test
    accs, rocs, clfs = stable_test(X, y, output_path="stable_test_result.npy")
    # predict
    y_preds = predict(clfs, X)

    # analyse
    rocsplot(rocs, output_path="ROC.pdf", show=True)
    comprehensive_result = get_comprehensive_comparison(
        accs, rocs, output_path="comprehensive_result.csv"
    )


@cli.command()
@click.option("--xlsx_path")
@click.option("--preprocess_func", default="default")
def feature_analyse(xlsx_path, preprocess_func):
    X, y, filter_data = load_xlsx(xlsx_path, preprocess_func)

    importance = cal_RF_feature_importance(
        X, y, filter_data.columns, output_path="RF_feature_importance_result_{}.csv"
    )


@cli.command()
@click.option("--xlsx_path")
@click.option("--test_xlsx_path")
@click.option("--preprocess_func", default="default")
def independent_verification(xlsx_path, test_xlsx_path, preprocess_func):
    X, y, _ = load_xlsx(xlsx_path, preprocess_func)
    X_trains, _, y_trains, _ = generate_datasets(X, y)
    clfs = train_models(X_trains, y_trains)

    X_tests, y_tests, _ = load_xlsx(test_xlsx_path, preprocess_func)
    accs, rocs = cal_accs_and_rocs(clfs, X_tests, y_tests, n_class=np.unique(y).size)

    # analyse
    rocsplot(rocs, output_path="ROC.pdf", show=True)
    comprehensive_result = get_comprehensive_comparison(
        accs, rocs, output_path="comprehensive_result.csv"
    )


if __name__ == "__main__":
    cli()
