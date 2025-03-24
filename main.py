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
    'gender': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    'body_type': [1, 2, 0, 1, 1, 0, 2, 1, 2, 0],
    'chest': [90, 80, 100, 85, 95, 78, 88, 92, 76, 105],
    'waist': [75, 65, 85, 70, 80, 62, 73, 78, 60, 90],
    'hip': [95, 85, 105, 90, 100, 82, 93, 98, 80, 110],
    'shoulder_width': [42, 38, 45, 40, 44, 36, 41, 43, 35, 48],
    'size': [1, 0, 2, 1, 2, 0, 1, 2, 0, 2]
}

df = pd.DataFrame(data)

# -------------------- Preprocessing --------------------
X = df[['height', 'weight', 'age
