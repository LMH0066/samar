import numpy as np
import pandas as pd

from samar.draw import lineplot


def rocsplot(rocs: dict, output_path: str = None):
    _rocs = pd.DataFrame()
    for method_name in rocs.keys():
        roc = pd.DataFrame(rocs[method_name])
        epoch_num = roc.shape[0]

        mean_fpr, tprs = np.linspace(0, 1, 100), []
        for _, row in roc.iterrows():
            _fpr, _tpr = row["fpr"], row["tpr"]
            interp_tpr = np.interp(mean_fpr, _fpr, _tpr)
            interp_tpr[0], interp_tpr[-1] = 0.0, 1
            tprs.extend(interp_tpr.tolist())

        _df = pd.DataFrame(
            {
                "FPR": mean_fpr.tolist() * epoch_num,
                "TPR": tprs,
                "method": "{}(AUC={})".format(method_name, round(roc["auc"].mean(), 2)),
            }
        )
        _rocs = pd.concat([_rocs, _df])
    _rocs.reset_index(drop=True, inplace=True)

    lineplot(
        _rocs,
        x="FPR",
        y="TPR",
        hue="method",
        output_path=output_path,
        figsize=(8, 8),
    )


def get_comprehensive_comparison(
    accs: dict, rocs: dict, output_path: str = None
) -> pd.DataFrame:
    acc = pd.DataFrame(accs)
    acc_result = pd.DataFrame(acc.mean(), columns=["Accuracy(%)"])
    acc_result = (round(acc_result * 100, 2)).astype(str)
    acc_result["Accuracy(%)"] += " ±" + (round(acc.std() * 100, 2)).astype(str)

    roc = pd.DataFrame(rocs)
    auc = roc.map(lambda x: x["auc"])
    auc_result = pd.DataFrame(auc.mean(), columns=["ROC-AUC"])
    auc_result = (round(auc_result, 2)).astype(str)
    auc_result["ROC-AUC"] += " ±" + (round(auc.std(), 2)).astype(str)

    comprehensive_result = pd.concat([acc_result, auc_result], axis=1)
    if output_path:
        comprehensive_result.to_csv(output_path)
    return comprehensive_result
