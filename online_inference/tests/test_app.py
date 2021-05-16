from fastapi.testclient import TestClient
import pytest
import pandas as pd
import numpy as np
from src.app import app

OK_STATUS_CODE = 200
ERROR_STATUS_CODE = 400


@pytest.fixture(scope='session')
def tmp_data():
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    return pd.DataFrame({feature: [value] for feature, value in zip(features, np.arange(len(features)))})


def test_request_correct_data(tmp_data):
    with TestClient(app) as client:
        data = tmp_data.values.tolist()
        features = tmp_data.columns.tolist()
        response = client.get("/predict/", json={"data": data, "features": features})
    assert response.status_code == OK_STATUS_CODE


def test_request_incorrect_data(tmp_data):
    with TestClient(app) as client:
        data = tmp_data.values.tolist()
        features = tmp_data.columns.tolist()
        response = client.get("/predict/", json={"data": data[1:], "features": features[1:]})
        assert response.status_code == ERROR_STATUS_CODE


def test_request_wide_data(tmp_data):
    tmp_data = tmp_data.assign(extra_column=lambda x: x.age)
    with TestClient(app) as client:
        data = tmp_data.values.tolist()
        features = tmp_data.columns.tolist()
        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == ERROR_STATUS_CODE


def test_request_shake_data(tmp_data):
    columns = tmp_data.columns.tolist()
    tmp_data = tmp_data[columns[1:] + columns[:1]]
    with TestClient(app) as client:
        data = tmp_data.values.tolist()
        features = tmp_data.columns.tolist()
        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == ERROR_STATUS_CODE


def test_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == OK_STATUS_CODE
