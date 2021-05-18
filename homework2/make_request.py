import logging
import requests
import json

from data import read_dataset


logger = logging.getLogger(__name__)


DATA_PATH = 'data/raw/data.csv'


if __name__ == "__main__":
    data = read_dataset(DATA_PATH)
    data_dict = data.to_dict(orient='records')
    
    for item in data_dict:
        response = requests.post(
            "http://0.0.0.0:8000/predict",
            json=item,
        )
        
        logger.info(f"Status code: {response.status_code}\n")
        logger.info(f"Response body: {response.json()}\n")
