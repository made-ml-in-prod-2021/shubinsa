import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_dataset():
    """ synthetic dataset generator """
    categorical_features = {'sex': 2, 'cp': 4, 'fbs': 2, 'restecg': 3, 'exang': 2, 'slope': 3, 'ca': 5, 'thal': 4}
    numerical_features = {
        'age': (29, 77), 'trestbps': (94, 200), 'chol': (126, 564),
        'thalach': (71, 202), 'oldpeak': (0, 6.2),
    }
    target = {'target': 2}

    size = 500
    data = dict()
    for feature, n_value in categorical_features.items():
        data[feature] = np.random.randint(0, n_value, size)
    for feature, (min_value, max_value) in numerical_features.items():
        data[feature] = np.random.randint(min_value, max_value + 1, size)
    for feature, n_value in target.items():
        data[feature] = np.random.randint(0, n_value, size)

    data = pd.DataFrame(data)
    return data



