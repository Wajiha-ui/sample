import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder

# Encode size labels into numbers
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)  # Convert test labels to numbers
y_pred_encoded = le.transform(y_pred)  # Convert predicted labels to numbers

# Compute MAE
mae = mean_absolute_error(y_test_encoded, y_pred_encoded)

# European Size Chart Data (Realistic Measurements)
data = {
    "height": [160, 165, 170, 175, 180, 185, 190, 195],
    "weight": [50, 55, 60, 70, 80, 90, 100, 110],
    "chest": [82, 86, 90, 94, 98, 102, 106, 110],
    "waist": [66, 70, 74, 78, 82, 86, 90, 94],
    "hips": [86, 90, 94, 98, 102, 106, 110, 114],
    "shoulder": [40, 42, 44, 46, 48, 50, 52, 54],
    "size": ["XS", "S", "M", "L", "XL", "XXL", "XXXL", "XXXXL"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and Labels
X = df.drop(columns=["size"])
y = df["size"]

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
mae = mean_absolute_error(y_test, [df['size'].tolist().index(size) for size in y_pred])

# Streamlit UI
st.title("Personalized Clothing Size Recommendation")
st.write(f"✅ Accuracy: {accuracy:.2f}% | 📉 MAE: {mae:.2f}")

height = st.number_input("Enter your height (cm)", min_value=140, max_value=210, step=1)
weight = st.number_input("Enter your weight (kg)", min_value=40, max_value=150, step=1)
chest = st.number_input("Enter your chest size (cm)", min_value=70, max_value=130, step=1)
waist = st.number_input("Enter your waist size (cm)", min_value=60, max_value=120, step=1)
hips = st.number_input("Enter your hip size (cm)", min_value=70, max_value=140, step=1)
shoulder = st.number_input("Enter your shoulder width (cm)", min_value=35, max_value=60, step=1)

if st.button("Predict Size"):
    user_data = np.array([[height, weight, chest, waist, hips, shoulder]])
    user_data_scaled = scaler.transform(user_data)
    predicted_size = model.predict(user_data_scaled)[0]
    st.success(f"🛍 Recommended Size: {predicted_size}")
