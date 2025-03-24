import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# -------------------- Data Preparation --------------------
st.title("👕 AI-Powered Clothing Size Predictor")

# Sample dataset (This should ideally be expanded with real-world data)
data = {
    'height': [170, 160, 180, 165, 175, 158, 169, 172, 155, 185, 190, 178, 162, 177, 168, 182, 173, 159, 176, 161],
    'weight': [60, 50, 70, 55, 65, 48, 63, 68, 45, 75, 80, 72, 52, 67, 62, 78, 66, 49, 64, 53],
    'age': [25, 22, 28, 24, 27, 21, 26, 29, 20, 30, 35, 32, 23, 31, 27, 33, 26, 19, 28, 21],
    'gender': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],  # 0 = Male, 1 = Female
    'body_type': [1, 2, 0, 1, 1, 0, 2, 1, 2, 0, 0, 1, 2, 1, 0, 1, 1, 2, 0, 2],  # 0 = Ecto, 1 = Meso, 2 = Endo
    'size': [1, 0, 2, 1, 2, 0, 1, 2, 0, 2, 2, 2, 0, 1, 0, 2, 1, 0, 1, 0]  # 0 = S, 1 = M, 2 = L
}

df = pd.DataFrame(data)

# Features & Target
X = df[['height', 'weight', 'age', 'gender', 'body_type']].copy()
y = df['size']

# Standardize numerical features safely
scaler = StandardScaler()
X.loc[:, ['height', 'weight', 'age']] = scaler.fit_transform(X[['height', 'weight', 'age']])

# Train a Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=5, min_samples_split=3, min_samples_leaf=3, random_state=42)
model.fit(X, y)

# -------------------- Streamlit UI --------------------

# Input fields
st.subheader("Enter Your Details:")
height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170, step=1)
weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70, step=1)
age = st.number_input("Age", min_value=15, max_value=80, value=25, step=1)
gender = st.radio("Gender", ["Male", "Female"])
body_type = st.radio("Body Type", ["Ectomorph (Slim)", "Mesomorph (Athletic)", "Endomorph (Broad)"])

# Convert inputs to numerical values
gender_value = 0 if gender == "Male" else 1
body_type_mapping = {"Ectomorph (Slim)": 0, "Mesomorph (Athletic)": 1, "Endomorph (Broad)": 2}
body_type_value = body_type_mapping[body_type]

# Predict button
if st.button("🔍 Predict Clothing Size"):
    # Prepare input data
    user_input = pd.DataFrame([[height, weight, age, gender_value, body_type_value]],
                              columns=['height', 'weight', 'age', 'gender', 'body_type'])

    # Apply scaling
    user_input.loc[:, ['height', 'weight', 'age']] = scaler.transform(user_input[['height', 'weight', 'age']])

    # Make prediction
    predicted_size = model.predict(user_input)[0]

    # Size mapping
    size_mapping = {0: "S", 1: "M", 2: "L"}
    st.success(f"🎯 Recommended Size: **{size_mapping[predicted_size]}**")
