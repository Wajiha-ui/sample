import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# -------------------- Expanded Sample Data (More Flexibility) --------------------
data = {
    'height': list(range(150, 201, 5)),  # Heights from 150 cm to 200 cm
    'weight': list(range(40, 121, 5)),  # Weights from 40 kg to 120 kg
    'age': list(range(18, 61, 5)),  # Ages from 18 to 60
    'gender': [0, 1] * 10,  # Alternating Male/Female
    'body_type': [0, 1, 2] * 7,  # Slim, Athletic, Curvy
    'chest': list(range(30, 131, 5)),  # Chest sizes from 30 to 130 cm
    'waist': list(range(30, 111, 5)),  # Waist sizes from 30 to 110 cm
    'hip': list(range(30, 131, 5)),  # Hip sizes from 30 to 130 cm
    'shoulder_width': list(range(35, 56, 5)),  # Shoulder width from 35 to 55 cm
    'size': [0, 1, 2] * 7  # Small, Medium, Large
}

df = pd.DataFrame(data)

# -------------------- Preprocessing --------------------
X = df.iloc[:, :-1]
y = df['size']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Updated Model (More Generalized) --------------------
model = RandomForestClassifier(
    n_estimators=50,  
    max_depth=6,  
    min_samples_split=5,  
    random_state=42
)
model.fit(X_train, y_train)

# -------------------- Accuracy Testing --------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# -------------------- Streamlit UI --------------------
st.title("👕 Personalized Clothing Size Recommendation")

st.write("Enter your details to get the best size recommendation:")

height = st.slider("Height (cm)", 150, 200, 170)
weight = st.slider("Weight (kg)", 40, 120, 70)
age = st.slider("Age", 18, 60, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
body_type = st.selectbox("Body Type", ["Slim", "Athletic", "Curvy"])
chest = st.selectbox("Chest (cm)", list(range(30, 131, 5)))
waist = st.selectbox("Waist (cm)", list(range(30, 111, 5)))
hip = st.selectbox("Hip (cm)", list(range(30, 131, 5)))
shoulder_width = st.slider("Shoulder Width (cm)", 35, 55, 45)

# Convert user input to model format
gender_val = 1 if gender == "Male" else 0
body_type_val = {"Slim": 0, "Athletic": 1, "Curvy": 2}[body_type]

input_data = np.array([[height, weight, age, gender_val, body_type_val, chest, waist, hip, shoulder_width]])
input_data = scaler.transform(input_data)

# Predict size
predicted_size = model.predict(input_data)[0]
size_mapping = {0: "S", 1: "M", 2: "L"}
recommended_size = size_mapping[predicted_size]

st.subheader(f"🛍 Recommended Size: **{recommended_size}**")
st.write(f"📊 Accuracy: **{accuracy * 100:.2f}%** | 📉 Mean Absolute Error: **{mae:.2f}**")
