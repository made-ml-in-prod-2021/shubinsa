import os
import pickle
import pandas as pd
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestClassifier
from ml_project.models.model_fit_predict import train_model, save_model
from ml_project.entities import TrainingParams
from typing import Tuple


def test_save_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators)
    real_output = save_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)
    assert model.predict(features).shape[0] == target.shape[0]


