from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_iris():
    response = client.post("/predict/", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert response.status_code == 200
    assert response.json() == {'predicted_class_name': 'versicolor', 'predicted_class_value': 1}

def test_predict_iris_bad_request():
    response = client.post("/predict/", json={"sepal_length": "wrong", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert response.status_code == 422  # Validation error
