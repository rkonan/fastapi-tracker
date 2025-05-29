
# Imports librairies
#from mlflow import MlflowClient
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import logging
from fastapi_tracker import FastAPITracker
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)
# Set tracking experiment


# Define experiment name, run name and artifact_path name
apple_experiment =("Apple_Models_4")
run_name = "new_run_2"
artifact_path = "rf_apples"

tracker = FastAPITracker("http://attowla.duckdns.org/tracker-api")
tracker.set_experiment(apple_experiment)

# Import Database
import os

file_path = os.path.join(os.path.dirname(__file__), "data", "fake_data.csv")
data = pd.read_csv(file_path)
#data = pd.read_csv("/kaggle/input/fake-data/fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')
y = data["demand"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
params = {
    "n_estimators": 20,
    "max_depth": 15,
    "random_state": 42,
}
rf = RandomForestRegressor(**params)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


with tracker.start_run(run_name=run_name) as run:
      tracker.log_params(params)
      tracker.log_metrics(metrics)
      tracker.log_model(rf, artifact_path="rf_apples", model_type="sklearn")
#      tracker.log_artifact("data/fake_data.csv","data")