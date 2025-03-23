import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

st.title("🔥 Personal Fitness Tracker 🔥")
st.write("Enter your details to predict calories burned during exercise.")

# Sidebar Inputs
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    height = st.sidebar.slider("Height (cm)", 100, 220, 170)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
    duration = st.sidebar.slider("Duration (min)", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 200, 100)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender = st.sidebar.radio("Gender", ("Male", "Female"))
    gender_encoded = 1 if gender == "Male" else 0
    bmi = round(weight / ((height / 100) ** 2), 2)
    return pd.DataFrame({"Age": [age], "BMI": [bmi], "Duration": [duration], "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender_encoded]})

df = user_input_features()
st.write("### Your Entered Data:", df)

# Load & Merge Data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
data = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)

data = pd.get_dummies(data, drop_first=True)
X = data.drop("Calories", axis=1)
y = data["Calories"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=42)
model.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
calorie_prediction = model.predict(df)[0]
st.write("### Predicted Calories Burned:", round(calorie_prediction, 2), "kcal")

# Visualization
st.write("---")
st.write("### Data Distribution")
fig, ax = plt.subplots()
sns.histplot(data["Calories"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.write("---")
st.write("### Similar Cases")
similar_cases = data[(data["Calories"] >= calorie_prediction - 10) & (data["Calories"] <= calorie_prediction + 10)]
st.write(similar_cases.sample(5))

# Deployment Instructions
st.write("---")
st.write("### Deployment Steps")
st.code("streamlit run app.py", language="bash")
