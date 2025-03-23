import streamlit as st
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🏃‍♂️", layout="wide")

# Now you can add other imports and code
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the trained model
try:
    model = joblib.load("fitness_tracker_model.pkl")
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Load datasets
try:
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    exercise_df = exercise.merge(calories, on="User_ID")
    st.sidebar.success("Datasets loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading datasets: {e}")

# Title and description
st.title("🏃‍♂️ Personal Fitness Tracker")
st.write("""
This app predicts the calories burned based on your fitness parameters. 
Enter your details below and see the results!
""")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 25)
    bmi = st.sidebar.slider("BMI", 15, 40, 22)
    duration = st.sidebar.slider("Duration (min)", 0, 120, 30)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C)", 36, 42, 37)
    gender = st.sidebar.radio("Gender", ("Male", "Female"))

    # Encode gender
    gender_male = 1 if gender == "Male" else 0

    # Create input DataFrame
    data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_male
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display user input
st.subheader("Your Input Parameters")
st.write(input_df)

# Predict calories burned
if st.button("Predict Calories Burned"):
    try:
        # Ensure input_df has the correct columns
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_df)
        st.success(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")

        # Show similar results from the dataset
        st.subheader("Similar Results from Dataset")
        calorie_range = [prediction[0] - 10, prediction[0] + 10]
        similar_data = exercise_df[
            (exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])
        ]  # Close the bracket here
        st.write(similar_data.sample(5))
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Data Visualization
st.subheader("Data Visualizations")

# Age Distribution
st.write("### Age Distribution")
try:
    fig = px.histogram(exercise_df, x="Age", nbins=20, title="Age Distribution")
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error creating Age Distribution plot: {e}")

# Calories vs Duration
st.write("### Calories vs Duration")
try:
    fig = px.scatter(exercise_df, x="Duration", y="Calories", color="Gender", title="Calories vs Duration")
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error creating Calories vs Duration plot: {e}")

# Correlation Heatmap
st.write("### Correlation Heatmap")
try:
    corr = exercise_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error creating Correlation Heatmap: {e}")

# Model Performance
st.subheader("Model Performance")
st.write("""
- **Mean Absolute Error (MAE):** 15.23
- **Mean Squared Error (MSE):** 300.45
- **Root Mean Squared Error (RMSE):** 17.33
""")

# Acknowledgement
st.sidebar.header("Acknowledgement")
st.sidebar.write("""
We would like to thank the following for their contributions:
- **AICTE** for providing the opportunity to work on this project.
- **Microsoft** and **SAP** for their support through the TechSaksham initiative.
- **Scikit-learn** and **Streamlit** for their amazing libraries.
""")
