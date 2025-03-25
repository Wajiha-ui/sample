import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load or create a sample dataset (replace with real European size data later)
data = {
    "height": np.random.randint(150, 200, 500),
    "weight": np.random.randint(45, 120, 500),
    "age": np.random.randint(18, 65, 500),
    "chest": np.random.randint(30, 60, 500),
    "waist": np.random.randint(20, 50, 500),
    "hips": np.random.randint(30, 60, 500),
    "shoulder": np.random.randint(20, 50, 500),
    "size": np.random.choice(["XS", "S", "M", "L", "XL", "XXL"], 500)
}

df = pd.DataFrame(data)

# Split dataset
X = df.drop(columns=["size"])
y = df["size"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred) * 100
mae = mean_absolute_error(pd.factorize(y_test)[0], pd.factorize(y_pred)[0])

# Streamlit UI
st.title("Personalized Clothing Size Recommendation")
st.write(f"✅ Accuracy: {accuracy:.2f}% | 📉 MAE: {mae:.2f}")

# User input
height = st.slider("Height (cm)", 150, 200, 170)
weight = st.slider("Weight (kg)", 45, 120, 70)
age = st.slider("Age", 18, 65, 30)
chest = st.slider("Chest (cm)", 30, 60, 40)
waist = st.slider("Waist (cm)", 20, 50, 30)
hips = st.slider("Hips (cm)", 30, 60, 40)
shoulder = st.slider("Shoulder Width (cm)", 20, 50, 35)

# Make prediction
input_data = np.array([[height, weight, age, chest, waist, hips, shoulder]])
input_data_scaled = scaler.transform(input_data)
predicted_size = rf_model.predict(input_data_scaled)[0]

st.subheader(f"🛍 Recommended Size: {predicted_size}")
