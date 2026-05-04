import pandas as pd
import json
import joblib


train = pd.read_csv("data/training_data.csv")
live = pd.read_csv("data/new_data.csv")


model = joblib.load("models/best_model.pkl")


predictions = model.predict(
    live[[
        "trip_distance_km",
        "vehicle_age_years",
        "load_weight_tons",
        "route_type"
    ]]
)


trip_train = train["trip_distance_km"].mean()
trip_live = live["trip_distance_km"].mean()

load_train = train["load_weight_tons"].mean()
load_live = live["load_weight_tons"].mean()


trip_shift = abs(trip_live - trip_train)
load_shift = abs(load_live - load_train)


alerts = []


if trip_shift > 135.53:
    alerts.append({
        "feature": "trip_distance_km",
        "train_mean": round(float(trip_train), 2),
        "live_mean": round(float(trip_live), 2),
        "shift": round(float(trip_shift), 2),
        "threshold": 135.53,
        "status": "ALERT"
    })


if load_shift > 2.66:
    alerts.append({
        "feature": "load_weight_tons",
        "train_mean": round(float(load_train), 2),
        "live_mean": round(float(load_live), 2),
        "shift": round(float(load_shift), 2),
        "threshold": 2.66,
        "status": "ALERT"
    })


result = {
    "total_predictions": 50,
    "mean_prediction": float(predictions.mean()),
    "drift_detected": len(alerts) > 0,
    "alerts": alerts
}


with open("results/step3_s5.json", "w") as f:
    json.dump(result, f, indent=2)


print(result)