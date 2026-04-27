from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from noshow_iq.model import predict
from noshow_iq.preprocess import preprocess, get_features_and_target
import pymongo
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NoShowIQ")

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = pymongo.MongoClient(MONGO_URI)
db = client["noshow_iq"]
predictions_col = db["predictions"]
training_runs_col = db["training_runs"]


# Input schema
class AppointmentInput(BaseModel):
    age: int
    scholarship: int
    hypertension: int
    diabetes: int
    alcoholism: int
    handicap: int
    sms_received: int
    days_in_advance: int
    appointment_weekday: int


# ─── Routes ───────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc)}


@app.post("/predict")
def make_prediction(data: AppointmentInput):
    features = data.model_dump()

    # Get prediction
    result = predict(features)

    # Save to MongoDB
    doc = {
        "timestamp": datetime.now(timezone.utc),
        "input": features,
        "risk_level": result["risk_level"],
        "probability": result["probability"],
        "recommendation": result["recommendation"]
    }
    predictions_col.insert_one(doc)

    return result


@app.get("/history")
def history():
    docs = list(
        predictions_col.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(20)
    )
    return {"predictions": docs}


@app.get("/stats")
def stats():
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$risk_level", "HIGH"]}, 1, 0
                        ]
                    }
                },
                "low_risk_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$risk_level", "LOW"]}, 1, 0
                        ]
                    }
                },
                "average_probability": {"$avg": "$probability"}
            }
        }
    ]

    result = list(predictions_col.aggregate(pipeline))

    # Get last trained time
    last_run = training_runs_col.find_one(
        {}, {"_id": 0}, sort=[("timestamp", -1)]
    )

    if result:
        stats_data = result[0]
        stats_data.pop("_id", None)
        stats_data["average_probability"] = round(
            stats_data["average_probability"], 4
        )
        stats_data["last_trained"] = (
            last_run["timestamp"] if last_run else None
        )
        return stats_data

    return {
        "total_predictions": 0,
        "high_risk_count": 0,
        "low_risk_count": 0,
        "average_probability": 0,
        "last_trained": None
    }
