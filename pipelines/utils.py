import logging
import numpy as np
import pandas as pd
# from scipy.special import result

from src.data_cleaning import DataPreprocessingStrategy, DataCleaning

def get_data_for_test():
    try:
        df = pd.read_csv("data/heart.csv")
        df = df.sample(n=100)
        process_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df,process_strategy)
        process_data = data_cleaning.handle_data()
        df.drop(['target'], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(f"{e}")
        raise e