import click
import numpy as np
import pandas as pd

import samar
import samar.SSNHL
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
def predict_SSNHL(xlsx_path):
    data = pd.read_excel(xlsx_path)
    data["Left ear/Right ear"] = data["Left ear/Right ear"].replace(
        ["L", "R", "L/R"], [1, 2, 3]
    )
    data["days from onset to treatment (categorized)"] = data[
        "days from onset to treatment (categorized)"
    ].replace(["≤7 days", "8-14 days", ">14days"], [0, 1, 2])
    data["Curve type (affected side)"] = data["Curve type (affected side)"].replace(
        ["normal", "downsloping", "flat", "profound", "upsloping"], [0, 1, 2, 3, 4]
    )
    data["Curve type (contralateral)"] = data["Curve type (contralateral)"].replace(
        ["normal", "downsloping", "flat", "profound", "upsloping"], [0, 1, 2, 3, 4]
    )
    return samar.SSNHL.predict(data.values[:, 1:-1])

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
