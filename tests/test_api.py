import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from noshow_iq.api import app

client = TestClient(app)

SAMPLE_INPUT = {
    "age": 30,
    "scholarship": 0,
    "hypertension": 0,
    "diabetes": 0,
    "alcoholism": 0,
    "handicap": 0,
    "sms_received": 1,
    "days_in_advance": 5,
    "appointment_weekday": 2
}

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_returns_ok():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"

@patch("noshow_iq.api.predictions_col")
def test_predict_returns_200(mock_col):
    mock_col.insert_one = MagicMock()
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200

@patch("noshow_iq.api.predictions_col")
def test_predict_returns_correct_fields(mock_col):
    mock_col.insert_one = MagicMock()
    response = client.post("/predict", json=SAMPLE_INPUT)
    data = response.json()
    assert "risk_level" in data
    assert "probability" in data
    assert "recommendation" in data

@patch("noshow_iq.api.predictions_col")
def test_predict_risk_level_valid(mock_col):
    mock_col.insert_one = MagicMock()
    response = client.post("/predict", json=SAMPLE_INPUT)
    data = response.json()
    assert data["risk_level"] in ["HIGH", "LOW"]

@patch("noshow_iq.api.predictions_col")
def test_predict_probability_range(mock_col):
    mock_col.insert_one = MagicMock()
    response = client.post("/predict", json=SAMPLE_INPUT)
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0

def test_predict_invalid_input():
    response = client.post("/predict", json={"age": "wrong"})
    assert response.status_code == 422

@patch("noshow_iq.api.predictions_col")
def test_history_returns_200(mock_col):
    mock_col.find.return_value.sort.return_value.limit.return_value = []
    response = client.get("/history")
    assert response.status_code == 200
