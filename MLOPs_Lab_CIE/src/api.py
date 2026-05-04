from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib


app = FastAPI()

model = joblib.load("models/best_model.pkl")


class TripInput(BaseModel):
    trip_distance_km: float = Field(..., ge=10, le=500)
    vehicle_age_years: int = Field(..., ge=1, le=15)
    load_weight_tons: float = Field(..., ge=0.5, le=10)
    route_type: int = Field(..., ge=1, le=3)


@app.get("/heartbeat")
def heartbeat():
    return {
        "status": "operational",
        "service": "FleetTrack API"
    }


@app.post("/estimate")
def estimate(data: TripInput):

    values = [[
        data.trip_distance_km,
        data.vehicle_age_years,
        data.load_weight_tons,
        data.route_type
    ]]

    prediction = model.predict(values)[0]

    return {
        "prediction": float(prediction)
    }