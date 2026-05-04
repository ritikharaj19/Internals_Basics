import json


result = {
    "model_name": "fleettrack-fuel-cost",
    "version": "v1",
    "stage": "Production",
    "transition_reason": "Lowest RMSE model selected"
}


with open("results/step4_s7.json", "w") as f:
    json.dump(result, f, indent=2)


print(result)