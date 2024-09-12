import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.evaluate import MSE, R2, RMSE
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[Annotated[float, "r2"], Annotated[float, "rmse"]]:
    """Evaluate model on the dataset

    Args:
        y_test: testing labels
        x_test: testing features
        model: RegressorMixin
    """
    try:
        prediction = model.predict(x_test)

        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2", r2)
        return r2, rmse
    except Exception as e:
        logging.error(f"Error in evaluation :- \n{e}")
        raise e