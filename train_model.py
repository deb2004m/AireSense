
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Dataset/Dataset.csv")

# Remove rows with missing pollutant values
df = df[df['pollutant_avg'].notna()]

# Pivot the table to have pollutants as columns
pivot_df = df.pivot_table(
    index=['country', 'state', 'city', 'station', 'last_update', 'latitude', 'longitude'],
    columns='pollutant_id',
    values='pollutant_avg'
).reset_index()

# Reset column names (pivot_table creates MultiIndex sometimes)
pivot_df.columns.name = None

# List of required pollutants
pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'OZONE']
filtered_df = pivot_df.dropna(subset=pollutants)

# Create AQI (simple mean)
filtered_df['AQI'] = filtered_df[pollutants].mean(axis=1)

# Features & target
X = filtered_df[pollutants]
y = filtered_df['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("📊 Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# Save model & data for UI
with open("aqi_model.pkl", "wb") as f:
    pickle.dump(model, f)

filtered_df.to_csv("Dataset/area_pollution_data.csv", index=False)

print("✅ Model trained and saved as aqi_model.pkl")

