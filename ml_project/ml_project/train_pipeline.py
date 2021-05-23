import sys
import logging
import logging.config
import yaml
import click
from ml_project.constants import *
from ml_project.entities import TrainingPipelineParams, read_training_pipeline_params
from ml_project.data import read_data, split_to_train_val
from ml_project.models import (
    train_model, predict_model, evaluate_model, save_model, save_metrics, load_model
)

from ml_project.features import FeaturesExtractor, extract_target


logger = logging.getLogger(APPLICATION_NAME)


def train_pipeline(params: TrainingPipelineParams):
    logger.info(f"start training pipeline - data reading from {params.input_data_path}")
    data = read_data(params.input_data_path)

    logger.info(f"data splitting to train and val")
    train_df, val_df = split_to_train_val(data, params.splitting_params)
    logger.debug(f"data splitted; train_df size: {train_df.shape}, val_df size: {val_df.shape}")

    feature_extractor = FeaturesExtractor(params.feature_params)
    X_train = feature_extractor.fit_transform(train_df)
    X_val = feature_extractor.transform(val_df)
    y_train = extract_target(train_df, params.feature_params)
    y_val = extract_target(val_df, params.feature_params)
    logger.info("features and target extracted")

    model = train_model(X_train, y_train, params.train_params)
    logger.info(f"model {params.train_params.model_type} loaded")

    y_predict = predict_model(model, X_val)
    logger.debug(f"prediction done; y_predict size: {y_predict.shape}")

    metrics = evaluate_model(y_predict, y_val)
    logger.info(f"evaluation done; accuracy: {metrics['accuracy_score']}")

    path_to_model = save_model(model, params.output_model_path)
    logger.info(f"model saved to {path_to_model}")
    save_metrics(metrics, params.metric_path)
    logger.info(f"metrics saved to {params.metric_path}")

    return path_to_model, metrics


def _setup_logging():
    handler = logging.StreamHandler(sys.stderr)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def setup_logging():
    """ setting up the logging from yaml config file """
    with open(DEFAULT_LOGGING_CONFIG_PATH) as config_fin:
        config = yaml.safe_load(config_fin)
        logging.config.dictConfig(config)


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    setup_logging()
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()


