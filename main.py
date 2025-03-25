import os

# Ensure xgboost is installed
os.system("pip install xgboost")

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ This should now work
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# -------------------- ⚡ Fast Data Loading --------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    num_samples = 10000  # Less data = Faster training
    data = {
        'height': np.random.normal(175, 10, num_samples),
        'weight': np.random.normal(75, 15, num_samples),
        'age': np.random.randint(18, 60, num_samples),
        'gender': np.random.choice([0, 1], num_samples),
        'body_type': np.random.choice([0, 1, 2], num_samples),
        'chest': np.random.normal(95, 15, num_samples),
        'waist': np.random.normal(85, 12, num_samples),
        'hip': np.random.normal(95, 12, num_samples),
        'shoulder_width': np.random.normal(45, 5, num_samples),
        'size': np.random.choice([0, 1, 2], num_samples, p=[0.33, 0.34, 0.33])
    }
    return pd.DataFrame(data)

df = load_data()

# -------------------- ⚡ Faster Model Training --------------------
X = df.iloc[:, :-1]
y = df['size']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔧 **XGBoost with Faster Settings**
model = xgb.XGBClassifier(
    n_estimators=100,  # ✅ Reduced from 500 to 100
    learning_rate=0.1,  # ✅ Faster convergence
    max_depth=5,  # ✅ Lower complexity = faster inference
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------- ✅ Fast Accuracy Calculation --------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# -------------------- ⚡ Streamlit UI --------------------
st.title("👕 AI Clothing Size Recommendation (Europe)")

st.write("Enter your details to get the best size recommendation:")

height = st.slider("Height (cm)", 150, 200, 175)
weight = st.slider("Weight (kg)", 40, 120, 75)
age = st.slider("Age", 18, 60, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
body_type = st.selectbox("Body Type", ["Slim", "Athletic", "Curvy"])
chest = st.slider("Chest (cm)", 20, 130, 95)
waist = st.slider("Waist (cm)", 20, 110, 85)
hip = st.slider("Hip (cm)", 30, 130, 95)
shoulder_width = st.slider("Shoulder Width (cm)", 20, 55, 45)

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
