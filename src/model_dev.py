import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, x_train, y_train):
        """
        Train the models
        Args:
            x_train:    training features
            y_train:    training targets

        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """

    def train(self, x_train, y_train, **kwargs):
        """
        Trains the model using Linear Regression
        Args:
            x_train:    features
            y_train:    targets
            **kwargs:

        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("Model Training Complete")
            return reg
        except Exception as e:
            logging.error(f"Error in Training Linear Regression Model :- \n{e}")
            raise e