import streamlit as st
import pickle
import numpy as np

# Load model, scaler, and feature list
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
selected_features = pickle.load(open("selected_features.pkl", "rb"))

st.title("💓 Heart Disease Prediction App")
st.write("Provide the patient info below to assess heart disease risk.")

# User inputs
Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
ExerciseAngina = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
Oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, 0.1)
ChestPainType = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA'])
ST_Slope_Flat = st.selectbox("ST Slope Flat (1 = Yes, 0 = No)", [0, 1])

# Manual one-hot encoding
ChestPainType_ATA = 1 if ChestPainType == 'ATA' else 0
ChestPainType_NAP = 1 if ChestPainType == 'NAP' else 0
ChestPainType_TA = 1 if ChestPainType == 'TA' else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[
        Sex, FastingBS, ExerciseAngina, Oldpeak,
        ChestPainType_ATA, ChestPainType_NAP,
        ChestPainType_TA, ST_Slope_Flat
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease ({round(probability * 100, 1)}%)")
    else:
        st.success(f"✅ Low risk of heart disease ({round((1 - probability) * 100, 1)}%)")
