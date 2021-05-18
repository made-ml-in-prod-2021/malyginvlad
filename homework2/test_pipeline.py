from fastapi.testclient import TestClient

from app import app
from data import read_dataset


DATA_PATH = 'data/raw/data.csv'

client = TestClient(app)


def test_start_server():
    
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "Hello, it is a first page for your predictions."
        

def test_predict():
    
    with TestClient(app) as client:
        test_data = read_dataset(DATA_PATH)
        data_dict = test_data.to_dict(orient='records')
        
        assert test_data.shape == (5, 13)

        for item in data_dict:
            response = client.post(
                "/predict",
                json=item,
            )

            assert response.status_code == 200

            result = response.json()
            assert len(result) == 1
            assert "prediction" in result
            assert result["prediction"] in [0, 1]

        
def test_validation_values():
    
    with TestClient(app) as client:
        test_data = read_dataset(DATA_PATH)
        data_dict = test_data.to_dict(orient='records')
        first_item = data_dict[0]
        
        first_item['sex'] = None
        response = client.post(
            "/predict",
            json=first_item,
        )
        assert response.status_code == 422