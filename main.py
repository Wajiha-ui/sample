import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset (replace with real data later)
data = {
    'height': [170, 160, 180, 165, 175, 158, 169, 172],
    'weight': [60, 50, 70, 55, 65, 48, 63, 68],
    'age': [25, 22, 28, 24, 27, 21, 26, 29],
    'gender': [1, 1, 0, 0, 1, 1, 0, 0],
    'body_type': [1, 2, 0, 1, 1, 0, 2, 1],
    'size': [1, 0, 2, 1, 2, 0, 1, 2]
}

df = pd.DataFrame(data)
X = df[['height', 'weight', 'age', 'gender', 'body_type']]
y = df['size']

# Train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Personalized Clothing Size Recommendation")

# User Inputs
height = st.number_input("Enter your height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Enter your weight (kg)", min_value=30, max_value=200, value=70)
age = st.number_input("Enter your age", min_value=10, max_value=100, value=25)
gender = st.radio("Select your gender", ("Male", "Female"))
body_type = st.selectbox("Select your body type", ["Ectomorph", "Mesomorph", "Endomorph"])

# Encoding inputs
gender = 0 if gender == "Male" else 1
body_type = {"Ectomorph": 0, "Mesomorph": 1, "Endomorph": 2}[body_type]

# Predict size
input_data = np.array([[height, weight, age, gender, body_type]])
predicted_size = model.predict(input_data)[0]

# Map prediction to size
size_mapping = {0: "S", 1: "M", 2: "L"}
# Ensure the predicted value stays within [0, 1, 2]
predicted_size = max(0, min(2, round(predicted_size)))
recommended_size = size_mapping[int(predicted_size)]


# Display result
st.write(f"Recommended Clothing Size: **{recommended_size}**")
