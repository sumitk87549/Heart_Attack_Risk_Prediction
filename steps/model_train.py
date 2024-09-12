import logging

import mlflow
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from zenml.client import  Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame,x_test: pd.DataFrame,y_train: pd.DataFrame,y_test: pd.DataFrame, config: ModelNameConfig):
# def train_model(x_train: pd.DataFrame,x_test: pd.DataFrame,y_train: pd.DataFrame,y_test: pd.DataFrame, config: ModelNameConfig) -> RegressorMixin:
    """Trains the model on the ingested data

    Args:
        x_train:    Train features
        x_test:     Test features
        y_train:    Train labels
        y_test:     Test labels
        config:
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(x_train,y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error in linear model training :- \n{e}")
        raise e