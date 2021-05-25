import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    """
    Open file csv.
    """
    data = pd.read_csv(path)
    
    return data
