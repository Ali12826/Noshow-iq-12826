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

MOCK_PREDICTION = {
    "risk_level": "HIGH",
    "probability": 0.82,
    "recommendation": "Call the patient to confirm."
}


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok():
    response = client.get("/health")
    assert response.json()["status"] == "ok"


@patch("noshow_iq.model.load_model")
@patch("noshow_iq.api.predictions_col")
def test_predict_returns_200(mock_col, mock_load):
    mock_load.return_value.predict_proba.return_value = [[0.18, 0.82]]
    mock_col.insert_one = MagicMock()
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200


@patch("noshow_iq.model.load_model")
@patch("noshow_iq.api.predictions_col")
def test_predict_returns_correct_fields(mock_col, mock_load):
    mock_load.return_value.predict_proba.return_value = [[0.18, 0.82]]
    mock_col.insert_one = MagicMock()
    response = client.post("/predict", json=SAMPLE_INPUT)
    data = response.json()
    assert "risk_level" in data
    assert "probability" in data
    assert "recommendation" in data


@patch("noshow_iq.model.load_model")
@patch("noshow_iq.api.predictions_col")
def test_predict_risk_level_valid(mock_col, mock_load):
    mock_load.return_value.predict_proba.return_value = [[0.18, 0.82]]
    mock_col.insert_one = MagicMock()
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.json()["risk_level"] in ["HIGH", "LOW"]


@patch("noshow_iq.model.load_model")
@patch("noshow_iq.api.predictions_col")
def test_predict_probability_range(mock_col, mock_load):
    mock_load.return_value.predict_proba.return_value = [[0.18, 0.82]]
    mock_col.insert_one = MagicMock()
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_predict_invalid_input():
    response = client.post("/predict", json={"age": "wrong"})
    assert response.status_code == 422


@patch("noshow_iq.api.predictions_col")
def test_history_returns_200(mock_col):
    mock_col.find.return_value.sort.return_value.limit.return_value = []
    response = client.get("/history")
    assert response.status_code == 200


@patch("noshow_iq.api.predictions_col")
@patch("noshow_iq.api.training_runs_col")
def test_stats_returns_200(mock_runs, mock_col):
    mock_col.aggregate.return_value = []
    mock_runs.find_one.return_value = None
    response = client.get("/stats")
    assert response.status_code == 200
