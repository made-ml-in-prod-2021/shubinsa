from typing import List
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline

from src.entities.dataclasses import AppResponse


def load_model(path: str) -> Pipeline:
    with open(path, "rb") as fin:
        return pickle.load(fin)


def make_predict(
    data: List, features: List[str], model: Pipeline,
):
    data = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)
    return [
        AppResponse(class_id=class_id) for class_id in predicts
    ]