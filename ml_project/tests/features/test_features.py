import os
import pytest
import numpy as np
import pandas as pd

from ml_project.entities import TrainingPipelineParams
from ml_project.entities import read_training_pipeline_params
from ml_project.features import FeaturesExtractor, extract_target


@pytest.fixture
def pipeline_params() -> TrainingPipelineParams:
    path = './tests/configs/train_config.yml'
    assert os.path.exists(path), 'choose another config path'
    params = read_training_pipeline_params(path)
    return params


@pytest.fixture
def feature_params(pipeline_params):
    fea_params = pipeline_params.feature_params
    return fea_params


@pytest.fixture
def extractor_instance(feature_params):
    extractor = FeaturesExtractor(feature_params)
    return extractor


def test_build_transformer(extractor_instance, synthetic_dataset):
    feature_params = extractor_instance.params
    transformer = extractor_instance.build_transformer()

    transformed = transformer.fit_transform(synthetic_dataset)
    expected_n_columns = (
        len(feature_params.numerical_features) + synthetic_dataset[feature_params.categorical_features].nunique().sum()
    )
    assert len(transformed) == len(synthetic_dataset)
    assert transformed.shape[1] == expected_n_columns


def test_transformation(extractor_instance, synthetic_dataset):
    extractor_instance.fit(synthetic_dataset)
    feature_params = extractor_instance.params
    transformed = extractor_instance.transform(synthetic_dataset)
    expected_n_columns = (
        len(feature_params.numerical_features) + synthetic_dataset[feature_params.categorical_features].nunique().sum()
    )
    assert len(transformed) == len(synthetic_dataset)
    assert transformed.shape[1] == expected_n_columns


def test_extract_target(feature_params, synthetic_dataset):
    target = extract_target(synthetic_dataset, feature_params)

    assert isinstance(target, pd.Series)
    assert target.shape == (len(synthetic_dataset), )
    assert np.all(synthetic_dataset[feature_params.target_col] == target)
