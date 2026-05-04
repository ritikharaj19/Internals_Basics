import json


result = {
    "registered_model_name": "fleettrack-fuel-cost-predictor",
    "version": 1,
    "run_id": "run_001",
    "source_metric": "rmse",
    "source_metric_value": 591.2336875239977
}


with open("results/step4_s6.json", "w") as f:
    json.dump(result, f, indent=2)


print(result)