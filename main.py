import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data/heart.csv')

# Load model & scaler
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Heart Disease Risk Prediction")

# Sidebar inputs
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", int(data['Age'].min()), int(data['Age'].max()), int(data['Age'].mean()))
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.sidebar.slider("Resting BP", int(data['Trestbps'].min()), int(data['Trestbps'].max()), int(data['Trestbps'].mean()))
chol = st.sidebar.slider("Cholesterol", int(data['Chol'].min()), int(data['Chol'].max()), int(data['Chol'].mean()))
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0,1])
restecg = st.sidebar.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.sidebar.slider("Max Heart Rate", int(data['Thalach'].min()), int(data['Thalach'].max()), int(data['Thalach'].mean()))
exang = st.sidebar.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0,1])
oldpeak = st.sidebar.slider("ST depression", float(data['Oldpeak'].min()), float(data['Oldpeak'].max()), float(data['Oldpeak'].mean()))
slope = st.sidebar.selectbox("Slope (0-2)", [0,1,2])
ca = st.sidebar.selectbox("Major Vessels (0-3)", [0,1,2,3])
thal = st.sidebar.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1,2,3])

input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Prediction
input_scaled = scaler.transform([input_data])
risk_prob = model.predict_proba(input_scaled)[0][1]
risk_pred = model.predict(input_scaled)[0]

st.subheader("Prediction")
st.write(f"Heart Disease Risk: {risk_prob*100:.2f}%")
st.write(f"Predicted Class: {'Heart Disease' if risk_pred==1 else 'No Heart Disease'}")

# Feature Insights
st.subheader("Feature Insights")
fig, ax = plt.subplots(1,2, figsize=(12,4))
sns.histplot(data['Age'], bins=20, kde=True, ax=ax[0])
ax[0].set_title("Age Distribution")
sns.scatterplot(data=data, x='Chol', y='Thalach', hue='Target', palette='Set1', ax=ax[1])
ax[1].set_title("Cholesterol vs Max Heart Rate")
st.pyplot(fig)