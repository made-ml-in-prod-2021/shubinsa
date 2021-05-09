from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.entities import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_to_train_val(
        data: pd.DataFrame, params: SplittingParams,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ splitter function to train and validation dataset"""
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
