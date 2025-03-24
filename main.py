import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os

# -------------------- Load or Create User Data --------------------
USER_DATA_FILE = "user_feedback.json"

def load_user_feedback():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

def save_user_feedback(user_feedback):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(user_feedback, file)

user_feedback = load_user_feedback()

# -------------------- Sample Data --------------------
st.title("👕 AI-Powered Personalized Clothing Size Predictor")

data = {
    'height': [170, 160, 180, 165, 175, 158, 169, 172, 155, 185],
    'weight': [60, 50, 70, 55, 65, 48, 63, 68, 45, 75],
    'age': [25, 22, 28, 24, 27, 21, 26, 29, 20, 30],
    'gender': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],  # 0 = Male, 1 = Female
    'body_type': [1, 2, 0, 1, 1, 0, 2, 1, 2, 0],  # 0 = Ecto, 1 = Meso, 2 = Endo
    'chest': [90, 80, 100, 85, 95, 78, 88, 92, 76, 105],
    'waist': [75, 65, 85, 70, 80, 62, 73, 78, 60, 90],
    'hip': [95, 85, 105, 90, 100, 82, 93, 98, 80, 110],
    'shoulder_width': [42, 38, 45, 40, 44, 36, 41, 43, 35, 48],
    'size': [1, 0, 2, 1, 2, 0, 1, 2, 0, 2]  # 0 = S, 1 = M, 2 = L
}

df = pd.DataFrame(data)

# -------------------- Preprocessing --------------------
X = df[['height', 'weight', 'age', 'gender', 'body_type', 'chest', 'waist', 'hip', 'shoulder_width']]
y = df['size']

scaler = StandardScaler()
X.loc[:, ['height', 'weight', 'age', 'chest', 'waist', 'hip', 'shoulder_width']] = scaler.fit_transform(X[['height', 'weight', 'age', 'chest', 'waist', 'hip', 'shoulder_width']])

model = DecisionTreeClassifier(max_depth=5, min_samples_split=3, min_samples_leaf=3, random_state=42)
model.fit(X, y)

# -------------------- User Input --------------------
st.subheader("Enter Your Details for a Personalized Fit:")

height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170, step=1)
weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70, step=1)
age = st.number_input("Age", min_value=15, max_value=80, value=25, step=1)
gender = st.radio("Gender", ["Male", "Female"])
body_type = st.radio("Body Type", ["Ectomorph (Slim)", "Mesomorph (Athletic)", "Endomorph (Broad)"])
chest = st.number_input("Chest Size (cm)", min_value=70, max_value=130, value=90, step=1)
waist = st.number_input("Waist Size (cm)", min_value=50, max_value=120, value=75, step=1)
hip = st.number_input("Hip Size (cm)", min_value=70, max_value=140, value=95, step=1)
shoulder_width = st.number_input("Shoulder Width (cm)", min_value=30, max_value=60, value=42, step=1)
fit_preference = st.radio("Fit Preference", ["Tight", "Regular", "Loose"])

# Convert categorical inputs
gender_value = 0 if gender == "Male" else 1
body_type_mapping = {"Ectomorph (Slim)": 0, "Mesomorph (Athletic)": 1, "Endomorph (Broad)": 2}
body_type_value = body_type_mapping[body_type]

# Predict Clothing Size
if st.button("🔍 Predict Clothing Size"):
    user_input = pd.DataFrame([[height, weight, age, gender_value, body_type_value, chest, waist, hip, shoulder_width]],
                              columns=['height', 'weight', 'age', 'gender', 'body_type', 'chest', 'waist', 'hip', 'shoulder_width'])

    user_input.loc[:, ['height', 'weight', 'age', 'chest', 'waist', 'hip', 'shoulder_width']] = scaler.transform(user_input[['height', 'weight', 'age', 'chest', 'waist', 'hip', 'shoulder_width']])

    predicted_size = model.predict(user_input)[0]

    # Size Mapping & Fit Adjustments
    size_mapping = {0: "S", 1: "M", 2: "L"}
    recommended_size = size_mapping[predicted_size]

    # Adjust based on fit preference
    if fit_preference == "Tight" and recommended_size != "S":
        recommended_size = "S"
    elif fit_preference == "Loose" and recommended_size != "L":
        recommended_size = "L"

    st.success(f"🎯 Recommended Size: **{recommended_size}**")

    # Store feedback
    user_feedback[str(user_input.values.tolist())] = recommended_size
    save_user_feedback(user_feedback)

# -------------------- User Feedback --------------------
st.subheader("Did This Prediction Work for You?")
feedback = st.radio("Was the recommended size correct?", ["Yes", "No"])
if feedback == "No":
    correct_size = st.selectbox("What size did you actually need?", ["S", "M", "L"])
    if st.button("Submit Feedback"):
        user_feedback[str(user_input.values.tolist())] = correct_size
        save_user_feedback(user_feedback)
        st.success("✅ Your feedback has been saved! This will improve future recommendations.")

