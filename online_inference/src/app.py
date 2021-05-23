import logging
import logging.config
import sys
from typing import List, Optional
import yaml

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from sklearn.pipeline import Pipeline
from src.constants import *

from src.entities import (
    read_app_params,
    AppRequest, AppResponse,
)
from src.models import make_predict, load_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

logger = logging.getLogger(APPLICATION_NAME)

DEFAULT_CONFIG_PATH = "configs/app_config.yaml"
model: Optional[Pipeline] = None
app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_app_model():
    app_params = read_app_params("configs/app_config.yaml")
    logger.info("Start loading model")
    global model
    model = load_model(app_params.model_path)
    logger.info("Model loaded")


@app.get("/predict/", response_model=List[AppResponse])
def predict(request: AppRequest):
    return make_predict(request.data, request.features, model)


def setup_app():
    setup_logging()
    app_params = read_app_params(DEFAULT_CONFIG_PATH)
    logger.info(f"Running app on {app_params.host} with port {app_params.port}")
    uvicorn.run(app, host=app_params.host, port=app_params.port)


def setup_logging():
    """ setting up the logging from yaml config file """
    with open(DEFAULT_LOGGING_CONFIG_PATH) as config_fin:
        config = yaml.safe_load(config_fin)
        logging.config.dictConfig(config)


if __name__ == "__main__":
    setup_app()
