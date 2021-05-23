import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ml_project.entities import FeatureParams
from sklearn.base import TransformerMixin


class FeaturesExtractor(TransformerMixin):
    """ FeaturesExtractor Class """

    def __init__(self, feature_params: FeatureParams):
        self.params = feature_params

    def fit(self, df: pd.DataFrame):
        self._transformer = self.build_transformer()
        self._transformer.fit(df)
        return self

    def transform(self, df: pd.DataFrame):
        return self._transformer.transform(df)

    @staticmethod
    def build_categorical_pipeline() -> Pipeline:
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(
                missing_values=np.nan,
                strategy="most_frequent"
            )),
            ("ohe", OneHotEncoder()),
        ])
        return categorical_pipeline

    @staticmethod
    def build_numerical_pipeline() -> Pipeline:
        operations = [("impute", SimpleImputer(
            missing_values=np.nan,
            strategy="mean")
        )]
        num_pipeline = Pipeline(operations)
        return num_pipeline

    def build_transformer(self) -> ColumnTransformer:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    self.build_categorical_pipeline(),
                    self.params.categorical_features,
                ),
                (
                    "numerical_pipeline",
                    self.build_numerical_pipeline(),
                    self.params.numerical_features,
                ),
            ]
        )
        return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
