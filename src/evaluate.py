import logging
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import numpy as np


class Evaluate(ABC):
    """ Abstract class defining strategy to evaluate our model
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Caclculate score for model
        Args:
            y_true: True labels
            y_pred: predicted labels

        Returns:
            None
        """
        pass


class MSE(Evaluate):
    """
    Evaluate using Mean Squared Error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Scoring model using Mean Squared Error
        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE = {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE :- \n{e}")
            raise e


class R2(Evaluate):
    """
    Evaluate using R2 Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Scoring model using R2 Score
        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
        """
        try:
            logging.info("calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 = {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 :- \n{e}")
            raise e


class RMSE(Evaluate):
    """
    Evaluate using Root Mean Squared Error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Scoring model using Root Mean Squared Error
        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
        """
        try:
            logging.info("Calculating RMSE")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"RMSE = {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE :- \n{e}")





