import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# -------------------- 🔧 Improved Training Data --------------------
num_samples = 10000  # 🔼 Increased sample size for better accuracy

np.random.seed(42)  # Ensures reproducibility

data = {
    'height': np.random.randint(150, 200, num_samples),
    'weight': np.random.randint(40, 120, num_samples),
    'age': np.random.randint(18, 60, num_samples),
    'gender': np.random.choice([0, 1], num_samples),  # 0 = Female, 1 = Male
    'body_type': np.random.choice([0, 1, 2], num_samples),  # 0 = Slim, 1 = Athletic, 2 = Curvy
    'chest': np.random.randint(30, 130, num_samples),
    'waist': np.random.randint(20, 110, num_samples),  # 🔼 Starts from 20 now
    'hip': np.random.randint(30, 130, num_samples),
    'shoulder_width': np.random.randint(35, 55, num_samples),
    'size': np.random.choice([0, 1, 2], num_samples, p=[0.33, 0.34, 0.33])  # 🔧 Balanced size distribution
}

df = pd.DataFrame(data)  

# -------------------- 🔧 Model Training --------------------
X = df.iloc[:, :-1]  
y = df['size']  

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔧 Improved Model for Higher Accuracy
model = RandomForestClassifier(
    n_estimators=200,   # 🔼 More trees
    max_depth=12,       # 🔼 Increased depth
    min_samples_split=4,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------- 🔧 Accuracy Testing --------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# -------------------- Streamlit UI --------------------
st.title("👕 AI Clothing Size Recommendation")

st.write("Enter your details to get the best size recommendation:")

height = st.slider("Height (cm)", 150, 200, 170)
weight = st.slider("Weight (kg)", 40, 120, 70)
age = st.slider("Age", 18, 60, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
body_type = st.selectbox("Body Type", ["Slim", "Athletic", "Curvy"])
chest = st.slider("Chest (cm)", 30, 130, 90)
waist = st.slider("Waist (cm)", 20, 110, 75)  # 🔼 Starts from 20 now
hip = st.slider("Hip (cm)", 30, 130, 95)
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
