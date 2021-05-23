from tests.synthetic_dataset_generator import synthetic_dataset
import pytest
import pandas as pd

from ml_project.data.make_dataset import read_data
from ml_project.entities.params import FeatureParams
from ml_project.features.build_features import FeaturesExtractor, extract_target
from typing import List, Tuple


@pytest.fixture
def dataset_path(tmpdir, synthetic_dataset):
    dataset_fio = tmpdir.join('dataset.txt')
    synthetic_dataset.to_csv(dataset_fio, index=None)
    return dataset_fio


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return []


@pytest.fixture
def features_and_target(
        dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=None,
        target_col="target",
    )
    data = read_data(dataset_path)
    feature_extractor = FeaturesExtractor(params)
    features = feature_extractor.fit_transform(data)
    target = extract_target(data, params)
    return features, target

