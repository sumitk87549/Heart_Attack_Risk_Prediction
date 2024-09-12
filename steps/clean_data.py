import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreprocessingStrategy
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_data(df: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame,"x_train"],Annotated[pd.DataFrame,"x_test"],Annotated[pd.Series,"y_train"],Annotated[pd.Series,"y_test"]]:
    """
    Cleans data and divides it into training and testing data

    Args:
        df: Uncleaned dataset

    Returns:
        x_train:    Training features
        x_test:     Testing features
        y_train:    Training labels
        y_test:     Testing labels

    """
    try:
        process_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df,process_strategy)
        process_data = data_cleaning.handle_data()
        data_divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(process_data,data_divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning, preprocessing and train-test split completed successfully")
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logging.error(f"Error in cleaning data :- \n{e}")
        raise e