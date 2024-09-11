import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining data strategy for handling data
    """
    
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreprocessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame :
        """ 
        Preprocess data
        
        Args:
            data (pd.DataFrame): Dataset

        Raises:
            e: Error in Data preprocessing

        Returns:
            pd.DataFrame: processed data
        """
        logging.info("Preprocessing data") 
        try:
            data = data.select_dtypes(include=[np.number])
            return data
        except Exception as e:
            logging.error(f"Error in processing data :- \n{e}")
            raise e 
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy to split data in test and train
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Splitting data into train and test

        Args:
            data (pd.DataFrame): Dataset

        Raises:
            e: Error in train test split

        Returns:
            pd.DataFrame: splitted dataset
        """
        try:
            x = data.drop('output', axis=1)
            y = data['output']
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
            return x_train, x_test, y_train,y_test
        except Exception as e:
            logging.error(f"Error in dividing data :- \n{e}")
            raise e 
        
class DataCleaning:
    """
    Class for data processing and train-test split
    """
    def __init__(self, data:pd.DataFrame, strategy:DataStrategy):
        self.data = data
        self.strategy = strategy
        
    
