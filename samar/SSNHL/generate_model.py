import joblib
import pandas as pd

from samar.compute import generate_datasets, train_models
from samar.util import load_xlsx

data = pd.read_excel("data.xlsx", index_col=0, header=[0])

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

X, y, _ = load_xlsx(
    data,
    "miNNseq",
    "prognostic(no_recovery=0, minor_recovery=1, important_recovery=2, full_recovery=3)",
)
X_trains, X_tests, y_trains, y_tests = generate_datasets(X, y, 50, 0.3)
clf = train_models(
    X_trains,
    y_trains,
    "classification",
    1,
    {
        "RandomForest": {
            "class_path": {
                "classification": "sklearn.ensemble.RandomForestClassifier",
            },
            "kwargs": {
                "n_estimators": 500,
                "max_features": 0.25,
                "criterion": "entropy",
            },
        }
    },
)["RandomForest"][0]
joblib.dump(clf, "model.pkl")
