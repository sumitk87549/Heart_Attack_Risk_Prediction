import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from data_path
    """
    def __init__(self, data_path:str):
        """

        Args:
            data_path (str): path to the data
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Ingesting data from dataset

        Returns:
            Dataframe : ingested data in pandas Dataframe format 
        """
        logging.info(f'Ingesting data from - {self.data_path}')
        return pd.read_csv(self.data_path)
    
@step # def ingest_data(data_path:str) -> None:
def ingest_data(data_path:str) -> pd.DataFrame:
    """Ingesting data from datapath

    Args:
        data_path (str): location of data

    Raises:
        e: Exception

    Returns:
        pd.DataFrame: ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error while ingesting data :- \n{e}')
        raise e