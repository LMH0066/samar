import click
import numpy as np

from samar.analyse import get_comprehensive_comparison, rocsplot
from samar.compute import (
    cal_accs_and_rocs,
    cal_shap,
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
@click.option("--xlsx_path", type=str)
@click.option("--task", type=str, default="classification or regression")
@click.option("--preprocess_func", type=str, default="default")
def general_process(xlsx_path, task, preprocess_func):
    X, y, _ = load_xlsx(xlsx_path, preprocess_func)

    # stable test
    scores, clfs = stable_test(X, y, task, output_path="stable_test_result.npy")
    # predict
    y_preds = predict(clfs, X)

    # analyse
    if task == "classification":
        rocsplot(scores["rocs"], output_path="ROC.pdf", show=True)
    results = get_comprehensive_comparison(
        scores, output_path="comprehensive_result.csv"
    )


@cli.command()
@click.option("--xlsx_path", type=str)
@click.option("--task", type=str, default="classification or regression")
@click.option("--preprocess_func", type=str, default="default")
def feature_analyse(xlsx_path, task, preprocess_func):
    X, y, filter_data = load_xlsx(xlsx_path, preprocess_func)

    importance = cal_shap(X, y, task)


@cli.command()
@click.option("--xlsx_path", type=str)
@click.option("--test_xlsx_path", type=str)
@click.option("--task", type=str, default="classification or regression")
@click.option("--preprocess_func", type=str, default="default")
def independent_verification(xlsx_path, test_xlsx_path, task, preprocess_func):
    X, y, _ = load_xlsx(xlsx_path, preprocess_func)
    X_trains, _, y_trains, _ = generate_datasets(X, y)
    clfs = train_models(X_trains, y_trains, task)

    X_tests, y_tests, _ = load_xlsx(test_xlsx_path, preprocess_func)
    accs, rocs = cal_accs_and_rocs(clfs, X_tests, y_tests, n_class=np.unique(y).size)

    # analyse
    rocsplot(rocs, output_path="ROC.pdf", show=True)
    results = get_comprehensive_comparison(
        {"accs": accs, "rocs": rocs}, output_path="comprehensive_result.csv"
    )


if __name__ == "__main__":
    cli()
