import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with your actual dataset

# Ensure 'size' column exists
if 'size' not in df.columns:
    st.error("Dataset does not contain a 'size' column. Please check your data.")
    st.stop()

# Drop NaN values
df.dropna(inplace=True)

# Features & Target
X = df.drop(columns=['size'])  # Features (all columns except 'size')
y = df['size']  # Target variable (size)

# Encode the size labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Convert sizes to numeric labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model (RandomForest for better handling of non-linearity)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Convert predictions back to original labels
y_pred_labels = le.inverse_transform(y_pred)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Streamlit UI
st.title("Personalized Clothing Size Recommendation")
st.markdown(f"✅ **Accuracy:** {accuracy:.2%} | 📉 **MAE:** {mae:.2f}")

# Allow users to input their measurements
st.sidebar.header("Enter Your Measurements")
user_data = {}
for col in X.columns:
    user_data[col] = st.sidebar.number_input(f"Enter {col}", value=np.mean(df[col]))

# Convert input into DataFrame
user_df = pd.DataFrame([user_data])

# Predict size for user
predicted_size = model.predict(user_df)
predicted_label = le.inverse_transform(predicted_size)

st.write(f"### Recommended Size: **{predicted_label[0]}**")

