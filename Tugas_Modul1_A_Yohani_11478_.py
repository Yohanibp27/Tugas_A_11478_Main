import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

model_directory = r'D:\Tugas_A_11478'
# model_directory = r'C:\Users\HP\Downloads\Semester 7\PMDPM - A\Pert - 2 Introduction to Machine Learning & MLOps with Python (Praktekl)\Tugas_A_11478'

model_path = os.path.join(model_directory, 'rf_diabetes_model.pkl')

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        rf_model = loaded_model[0]

        st.title("Prediksi Diabetes")

        st.write("Aplikasi ini digunakan untuk membuat prediksi penyakit diabetes pada seseorang")

        pregnancies = st.slider("Pregnancies", min_value=0, max_value=17, step=1)
        glucose = st.slider("Glucose (mg/dl)", min_value=0.0, max_value=199.0, step=0.1)
        bloodPressure = st.slider("Blood Pressure (mmHg)", min_value=0, max_value=122, step=2)
        skinThickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=99, step=2)
        insulin = st.slider("Insulin (µU/mL)", min_value=0, max_value=846, step=10)
        bmi = st.slider("BMI", min_value=0.0, max_value=67.1, step=0.1)
        diabetesPredigreeFucntion = st.slider("Diabetes Predigree Function", min_value=0.07, max_value=2.42, step=0.1)
        age = st.slider("Age", min_value=21, max_value=81, step=1)

        input_data = [[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPredigreeFucntion, age]]

        if st.button("Prediksi!"):
            rf_model_prediction = rf_model.predict(input_data)
            outcome_names = {0: 'Tidak Diabetes', 1: 'Diabetes'}
            st.write(f"Orang tersebut diprediksi **{outcome_names[rf_model_prediction[0]]}** oleh **RF**")
    
    except Exception as e:
        st.write("Terjadi kesalahan: {e}")
else:
    print("File 'rf_diabetes_model.pkl' tidak ditemukan di direktori")