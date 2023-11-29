import numpy as np
import pandas as pd
from fancyimpute import KNN
from sklearn.ensemble import *
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC


FUNS = [
    # Boosting
    AdaBoostClassifier,
    GradientBoostingClassifier,
    # Bagging
    RandomForestClassifier,
    ExtraTreesClassifier,
    # Other
    SVC,
]


def load_data(file_path, preprocess_func: str):
    data = pd.read_excel(file_path, index_col=0, header=[0])

    if preprocess_func:
        result_col = "efficacy evaluation"
        data = data.dropna(subset=[result_col])
        y = data[result_col].copy()
        data = data.drop(["efficacy evaluation"], axis=1)

        filter_data = Preprocess().do(data, preprocess_func)
        X = np.array(filter_data)
        y = np.array(y[filter_data.index])
        return X, y, filter_data
    else:
        return data


class Preprocess:
    def __init__(self):
        self.function_map = {
            "default": self.default,
            "KNN": self.KNN,
        }

    def do(self, data: pd.DataFrame, func_name: str):
        filter_data = self.function_map[func_name](data)

        for column in filter_data.columns:
            filter_data[column] = pd.to_numeric(filter_data[column])

        return filter_data

    def _clean_row(self, data: pd.DataFrame, string: str) -> pd.DataFrame:
        return data[
            ~data.apply(lambda row: row.astype(str).str.contains(string).any(), axis=1)
        ]

    def default(self, data: pd.DataFrame) -> pd.DataFrame:
        filter_data = data.copy()

        filter_data = filter_data.dropna()
        return filter_data

    def KNN(self, data: pd.DataFrame) -> pd.DataFrame:
        filter_data = data.copy()

        filled_data = KNN(k=10).fit_transform(filter_data)
        filter_data[:] = filled_data

        return filter_data


def train_model(clf, X_train, y_train, X_test, y_test, n_class):
    clf.fit(X_train, y_train)

    # evaluate
    accuracy = clf.score(X_test, y_test)
    y_pred_proba = clf.predict_proba(X_test)

    if n_class == 2:
        fpr, tpr, thersholds = roc_curve(y_test, y_pred_proba[:, 1])
    else:
        # multi class
        y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))
        fpr, tpr, thersholds = roc_curve(y_test_one_hot.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)
    roc = [fpr, tpr, roc_auc]

    return accuracy, roc


def write_accs_file(path, accs):
    pd.DataFrame(accs).to_csv(path)

def read_accs_file(path):
    return pd.read_csv(path, index_col=0)

def write_rocs_file(path, rocs):
    np.save(path, rocs)

def read_rocs_file(path):
    return np.load(path, allow_pickle=True).item()
