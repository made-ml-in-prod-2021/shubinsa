import sys
import yaml
import click
import pandas as pd
from ml_project.entities import TrainingPipelineParams, read_training_pipeline_params
from ml_project.data import read_data
from ml_project.models import predict_model, load_model
from ml_project.features import FeaturesExtractor
import logging.config
from ml_project.constants import *


logger = logging.getLogger(APPLICATION_NAME)


def predict_pipeline(
        dataset_path: str,
        output_path: str,
        params: TrainingPipelineParams,
    ):

    model = load_model(params.output_model_path)

    logger.info(f"loaded model from {params.output_model_path} for prediction")
    df = read_data(dataset_path)
    logger.info(f"data readed from {dataset_path}")
    logger.debug(f"data size: {df.shape}")
    extracted_features = FeaturesExtractor(params.feature_params).fit_transform(df)
    logger.debug(f"features extracted; extracted_features size: {extracted_features.shape}")
    predict = predict_model(model, extracted_features)
    logger.info(f"prediction done; prediction size: {predict.shape}")
    pd.DataFrame(predict, columns=['target']).to_csv(output_path, index=False)


def _setup_logging():
    handler = logging.StreamHandler(sys.stderr)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def setup_logging():
    """ setting up the logging from yaml config file """
    with open(DEFAULT_LOGGING_CONFIG_PATH) as config_fin:
        config = yaml.safe_load(config_fin)
        logging.config.dictConfig(config)


@click.command(name="predict_pipeline")
@click.argument("config_path", default=DEFAULT_CONFIG_PATH)
@click.argument("dataset_path", default=DEFAULT_DATASET_PATH)
@click.argument("output_path", default=DEFAULT_OUTPUT_PATH)
def predict_pipeline_command(config_path: str, dataset_path: str, output_path: str):
    setup_logging()
    params = read_training_pipeline_params(config_path)
    predict_pipeline(dataset_path, output_path, params)


if __name__ == "__main__":
    predict_pipeline_command()


