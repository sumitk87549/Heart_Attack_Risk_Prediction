import logging
import pandas as pd
from zenml import step

@step
def clean_data(df: pd.DataFrame) -> None:
# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    """Cleans the ingested data

    Args:
        df (pd.DataFrame): Unprocessed data

    Returns:
        pd.DataFrame: processed/cleaned data
    """
    pass