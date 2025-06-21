import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb  # Using XGBoost
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import random

def generate_crime_weather_dataset(num_days=1000, crime_range=(10, 100), temp_range=(20, 40), rain_range=(0, 50)):
    """Generates a synthetic crime and weather dataset with time and location."""

    date_range = pd.date_range(start='2021-01-01', periods=num_days, freq='D')
    crime_counts = np.random.randint(crime_range[0], crime_range[1], size=num_days)
    temperatures = np.random.uniform(temp_range[0], temp_range[1], size=num_days)
    rainfall = np.random.uniform(rain_range[0], rain_range[1], size=num_days)
    hours = np.random.randint(0, 24, size=num_days)
    locations = [f"Location_{random.randint(1, 5)}" for _ in range(num_days)]

    df = pd.DataFrame({
        'Date': date_range,
        'crime_count': crime_counts,
        'Temperature': temperatures,
        'Rainfall': rainfall,
        'Hour': hours,
        'Location': locations
    })
    return df

# Generate the dataset
dataset = generate_crime_weather_dataset()
dataset.to_csv('synthetic_crime_weather_data.csv', index=False)
print("Synthetic dataset 'synthetic_crime_weather_data.csv' created.")

# --- Data Loading ---
try:
    merged_df = pd.read_csv('synthetic_crime_weather_data.csv')
except FileNotFoundError:
    print("Error: synthetic_crime_weather_data.csv not found.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Data Preprocessing ---
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df['day_of_week'] = merged_df['Date'].dt.dayofweek
merged_df['month'] = merged_df['Date'].dt.month
merged_df['year'] = merged_df['Date'].dt.year

location_dummies = pd.get_dummies(merged_df['Location'], prefix='Location')
merged_df = pd.concat([merged_df, location_dummies], axis=1)
merged_df = merged_df.drop('Location', axis=1)

# --- Model Training and Hyperparameter Tuning (XGBoost) ---
X = merged_df[['Temperature', 'Rainfall', 'day_of_week', 'month', 'year', 'Hour'] + [col for col in merged_df.columns if col.startswith('Location_')]]
y = merged_df['crime_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# --- Model Evaluation ---
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")

accuracy = 100 - (mae / np.mean(y_test)) * 100
print(f"Accuracy: {accuracy:.2f}%")

# --- Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Crime Counts', marker='o')
plt.plot(y_pred, label='Predicted Crime Counts', marker='x')
plt.xlabel("Test Data Index")
plt.ylabel("Crime Counts")
plt.title("Actual vs. Predicted Crime Counts")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(merged_df['Date'], merged_df['crime_count'], label='Actual Crime Count')
plt.title('Crime Count Over Time')
plt.xlabel('Date')
plt.ylabel('Crime Count')
plt.legend()
plt.grid(True)
plt.show()