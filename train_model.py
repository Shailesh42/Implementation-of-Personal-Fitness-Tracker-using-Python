import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load datasets
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge datasets on User_ID
exercise_df = exercise.merge(calories, on="User_ID")

# Drop duplicates based on User_ID
exercise_df.drop_duplicates(subset=['User_ID'], keep='last', inplace=True)

# Drop the User_ID column as it is no longer needed
exercise_df.drop(columns="User_ID", inplace=True)

# Add BMI column
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# Split the data into training and testing sets
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

# One-hot encode the Gender column
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and target variable
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the RandomForestRegressor
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=1)
random_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = random_reg.predict(X_test)
print("RandomForest Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, y_pred), 2))
print("RandomForest Mean Squared Error (MSE):", round(mean_squared_error(y_test, y_pred), 2))
print("RandomForest Root Mean Squared Error (RMSE):", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

# Save the trained model to a file
joblib.dump(random_reg, "fitness_tracker_model.pkl")
print("Model saved as 'fitness_tracker_model.pkl'")