import click
import numpy as np
import pandas as pd

from samar.draw import lineplot
from samar.util import read_accs_file, read_rocs_file


def rocsplot(rocs, output_dir=None):
    _rocs = pd.DataFrame()
    for method_name in rocs.keys():
        _xs, _ys = rocs[method_name][0], rocs[method_name][1]

        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i in range(len(_xs)):
            _x, _y = _xs[i], _ys[i]
            interp_tpr = np.interp(mean_fpr, _x, _y)
            interp_tpr[0], interp_tpr[-1] = 0.0, 1
            _tprs = interp_tpr.tolist()
            tprs.extend(_tprs)

        _df = pd.DataFrame(
            {
                "FPR": mean_fpr.tolist() * len(_xs),
                "TPR": tprs,
                "method": "{}(AUC={})".format(
                    method_name, round(np.mean(rocs[method_name][2]), 2)
                ),
            }
        )
        _rocs = pd.concat([_rocs, _df])
    _rocs.reset_index(drop=True, inplace=True)

    lineplot(
        _rocs,
        x="FPR",
        y="TPR",
        hue="method",
        output_path="{}/ROC.pdf".format(output_dir) if output_dir else None,
        figsize=(8, 8),
    )


def get_comprehensive_comparison(accs, rocs, output_dir=None):
    acc_result = pd.DataFrame(accs.mean(), columns=["Accuracy(%)"])
    acc_result = (round(acc_result * 100, 2)).astype(str)
    acc_result["Accuracy(%)"] += " ±" + (round(accs.std() * 100, 2)).astype(str)

    auc = pd.DataFrame([rocs[key][2] for key in rocs.keys()], index=rocs.keys()).T
    auc_result = pd.DataFrame(auc.mean(), columns=["ROC-AUC"])
    auc_result = (round(auc_result, 2)).astype(str)
    auc_result["ROC-AUC"] += " ±" + (round(auc.std(), 2)).astype(str)

    comprehensive_result = pd.concat([acc_result, auc_result], axis=1)
    if output_dir:
        comprehensive_result.to_csv(
            "{}/comprehensive_comparison.csv".format(output_dir)
        )
    return comprehensive_result


@click.command()
@click.option("--accs_path", help=".csv file path", type=str)
@click.option("--rocs_path", default=".npy file path", type=str)
@click.option("--output_dir", help="Folder path for results output", type=str)
def run(accs_path, rocs_path, output_dir):
    accs, rocs = read_accs_file(accs_path), read_rocs_file(rocs_path)

    rocsplot(rocs, output_dir)
    get_comprehensive_comparison(accs, rocs, output_dir)


if __name__ == "__main__":
    run()
