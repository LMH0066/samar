import click
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from samar.util import FUNS, load_data, train_model, write_accs_file, write_rocs_file


def calculate(X, y, epoch):
    accs, rocs = dict(), dict()
    for function in FUNS:
        accuracy, roc = [], [[], [], []]
        for random_state in range(1, epoch+1):
            if function is SVC:
                clf = function(probability=True, random_state=random_state)
            else:
                clf = function(n_estimators=100, random_state=random_state)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=random_state
            )
            _accuracy, _roc = train_model(
                clf, X_train, y_train, X_test, y_test, np.unique(y).size
            )

            accuracy.append(_accuracy)
            roc[0].append(_roc[0].tolist())
            roc[1].append(_roc[1].tolist())
            roc[2].append(_roc[2])

        accs[clf.__class__.__name__] = accuracy
        rocs[clf.__class__.__name__] = roc

    return accs, rocs


@click.command()
@click.option("--data_path", help=".xlsx file path", type=str)
@click.option("--output_dir", help="Folder path for results output", type=str)
@click.option("--preprocess_func", default="default", type=str)
@click.option("--epoch", default=50, type=int)
def run(data_path, output_dir, preprocess_func, epoch):
    X, y, _ = load_data(data_path, preprocess_func)

    accs, rocs = calculate(X, y, epoch)

    write_accs_file("{}/accuracy.csv".format(output_dir), accs)
    write_rocs_file("{}/ROC.npy".format(output_dir), rocs)


if __name__ == "__main__":
    run()
