import pytest
from flask import Flask
import sys
import os

sys.path.append(os.path.abspath("scripts"))
from api_baseline import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Sentiment API" in response.data

def test_predict_positive(client):
    response = client.post('/predict', json={"text": "This movie was amazing!"})
    assert response.status_code == 200
    data = response.get_json()

    assert "prediction" in data
    assert data["prediction"] in ["Positive", "Negative"]
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_predict_empty_input(client):
    response = client.post('/predict', json={"text": ""})
    assert response.status_code == 200  

