import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    
    data = pd.read_csv(path)
    
    return data
