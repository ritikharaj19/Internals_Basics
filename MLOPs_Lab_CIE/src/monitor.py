import pandas as pd
import json


train = pd.read_csv("data/training_data.csv")
new = pd.read_csv("data/new_data.csv")


train_mean = train["trip_distance_km"].mean()
new_mean = new["trip_distance_km"].mean()


shift = abs(new_mean - train_mean)


drift_detected = shift > 100


result = {
    "feature_name": "trip_distance_km",
    "training_mean": float(train_mean),
    "production_mean": float(new_mean),
    "absolute_shift": float(shift),
    "drift_detected": bool(drift_detected)
}


with open("results/step3_s6.json", "w") as f:
    json.dump(result, f, indent=2)


print(result)