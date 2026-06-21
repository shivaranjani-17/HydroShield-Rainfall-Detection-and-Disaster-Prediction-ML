import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix
)
import joblib

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------
df = pd.read_csv("rainfall_disaster_prediction_dataset.csv")

# ------------------------------------------
# ENCODING
# ------------------------------------------
le_location = LabelEncoder()
le_risk = LabelEncoder()

df["Location"] = le_location.fit_transform(df["Location"])
df["Disaster Risk"] = le_risk.fit_transform(df["Disaster Risk"])

# ------------------------------------------
# FEATURES & TARGETS
# ------------------------------------------
X = df[[
    "Location",
    "Temperature (°C)",
    "Humidity (%)",
    "Pressure (hPa)",
    "Wind Speed (km/h)"
]]

y_rain = df["Rainfall (mm)"]
y_risk = df["Disaster Risk"]

# ------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------
X_train, X_test, y_rain_train, y_rain_test = train_test_split(
    X, y_rain, test_size=0.2, random_state=42
)

_, _, y_risk_train, y_risk_test = train_test_split(
    X, y_risk, test_size=0.2, random_state=42
)

# ------------------------------------------
# TRAIN MODELS
# ------------------------------------------
rain_model = RandomForestRegressor()
risk_model = RandomForestClassifier()

rain_model.fit(X_train, y_rain_train)
risk_model.fit(X_train, y_risk_train)

# ------------------------------------------
# RAINFALL MODEL ACCURACY
# ------------------------------------------
rain_pred = rain_model.predict(X_test)

print("----- RAINFALL MODEL ACCURACY -----")
print("R² Score:", r2_score(y_rain_test, rain_pred))
print("MAE:", mean_absolute_error(y_rain_test, rain_pred))
print("MSE:", mean_squared_error(y_rain_test, rain_pred))

# ------------------------------------------
# DISASTER RISK MODEL ACCURACY
# ------------------------------------------
risk_pred = risk_model.predict(X_test)

print("\n----- DISASTER RISK MODEL ACCURACY -----")
print("Accuracy:", accuracy_score(y_risk_test, risk_pred))
print("\nClassification Report:\n", classification_report(y_risk_test, risk_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_risk_test, risk_pred))
