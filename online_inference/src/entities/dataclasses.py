from pydantic import BaseModel, conlist, validator
from typing import List, Union

from src.entities import read_features_params
from src.constants import *

MODEL_FEATURES = read_features_params(DEFAULT_FEATURES_CONFIG_PATH).features


class AppRequest(BaseModel):
    data: List[conlist(Union[float, int])]
    features: List[str]

    @validator('features')
    def features_validator(cls, features):
        if features == MODEL_FEATURES:
            return features
        else:
            raise ValueError("Wrong features")


class AppResponse(BaseModel):
    class_id: int
