import click
import requests
import logging
import logging.config
import sys
import yaml

import numpy as np
import pandas as pd

from src.entities import read_app_params
from src.constants import *

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

logger = logging.getLogger(APPLICATION_NAME)


def setup_app_url(app_config):
    app_params = read_app_params(app_config)
    return f"http://{app_params.host}:{app_params.port}/predict/"


def setup_logging():
    """ setting up the logging from yaml config file """
    with open(DEFAULT_LOGGING_CONFIG_PATH) as config_fin:
        config = yaml.safe_load(config_fin)
        logging.config.dictConfig(config)


@click.command()
@click.option("--dataset_path", default='data/test_heart.csv')
@click.option("--app_config", default='configs/app_config.yaml')
def make_predict(dataset_path, app_config):
    setup_logging()
    logger.info(f"Read data from {dataset_path}")
    data = pd.read_csv(dataset_path)

    request_url = setup_app_url(app_config)
    logger.info(f"Request URL is {request_url}")

    request_features = list(data.columns)
    logger.info(f"Request features are {request_features}")

    for i in range(data.shape[0]):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        logger.info(f'Request: {request_data}')
        response = requests.get(
            request_url,
            json={"data": [request_data], "features": request_features},
        )
        logger.info(f'Response code: {response.status_code}, body: {response.json()}')


if __name__ == "__main__":
    make_predict()
