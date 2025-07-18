import os

import joblib
import numpy as np


def predict(X: np.array):
    _model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    clf = joblib.load(_model_path)
    return clf.predict(X)
