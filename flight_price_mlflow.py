import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set MLflow URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("flight_price_prediction")

# Load and preprocess data
flights = pd.read_csv("/home/shubham-sharma/Downloads/Voyage_Analytics/MLOPS Project/Data/flights.csv")
flights['date'] = pd.to_datetime(flights['date'], format='%m/%d/%Y')

# Label encoding
le = LabelEncoder()
flights['from'] = le.fit_transform(flights['from'])
flights['to'] = le.fit_transform(flights['to'])
flights['agency'] = le.fit_transform(flights['agency'])
flights['flightType'] = le.fit_transform(flights['flightType'])

# Extract date features
flights['day'] = flights['date'].dt.day
flights['month'] = flights['date'].dt.month
flights['year'] = flights['date'].dt.year

# Drop unused columns
flights.drop(columns=['date', 'travelCode', 'userCode'], inplace=True)

# Features and target
X = flights[['day', 'month', 'year', 'flightType', 'to', 'from', 'distance']]
y = flights['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = xgb_model.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Accuracy: {accuracy}')

# MLflow logging
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("accuracy", accuracy)

    # Log model with signature and example
    signature = infer_signature(X_train, xgb_model.predict(X_train))
    input_example = X_train.iloc[:5]

    model_info = mlflow.xgboost.log_model(
        xgb_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

# Load model back
loaded_model = mlflow.xgboost.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)
print("Sample Predictions:", predictions[:5])
