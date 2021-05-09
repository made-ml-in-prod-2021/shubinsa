import json
import pickle
import numpy as np
import pandas as pd
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from ml_project.entities import TrainingParams


SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]


def choose_model(model_type: str, random_state: int) -> SklearnRegressionModel:
    model_types = {
        'LogisticRegression': LogisticRegression(
            C=0.5,
            fit_intercept=True,
            max_iter=1500,
            random_state=random_state,
        ),
        'RandomForestClassifier': RandomForestClassifier(
            criterion='gini',
            max_features='log2',
            min_samples_leaf=4,
            n_estimators=200,
            random_state=random_state,
        ),
    }
    model = model_types.get(model_type, None)
    if model is None:
        raise NotImplementedError
    return model


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_params: TrainingParams
) -> SklearnRegressionModel:
    model = choose_model(train_params.model_type, train_params.random_state)
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame,
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series):
    scores = {
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predicts),
    }
    return scores


def save_model(model: SklearnRegressionModel, path: str):
    """ save model to pickle file """
    with open(path, "wb") as fout:
        pickle.dump(model, fout)
    return path


def save_metrics(metrics: dict, path: str):
    """ save metrics to json """
    with open(path, 'w') as fout:
        json.dump(metrics, fout)


def load_model(path: str) -> SklearnRegressionModel:
    with open(path, 'rb') as fin:
        loaded_model = pickle.load(fin)
    return loaded_model
