import pandas as pd
import json
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


mlflow.set_experiment("fleettrack-fuel-cost")

df = pd.read_csv("data/training_data.csv")

X = df.drop("fuel_cost", axis=1)
y = df["fuel_cost"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

results = []

best_rmse = float("inf")
best_model = None
best_name = None

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        mlflow.set_tag("domain", "fleet_management")

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name


joblib.dump(best_model, "models/best_model.pkl")

final_result = {
    "experiment_name": "fleettrack-fuel-cost",
    "models": results,
    "best_model": best_name,
    "best_metric_name": "rmse",
    "best_metric_value": float(best_rmse)
}

with open("results/step1_s1.json", "w") as f:
    json.dump(final_result, f, indent=2)

print("Task 1 complete")