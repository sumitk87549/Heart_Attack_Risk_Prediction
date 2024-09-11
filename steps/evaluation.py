import logging
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """Evaluate model on the dataset

    Args:
        df (pd.Dataframe): ingested data
    """
    pass