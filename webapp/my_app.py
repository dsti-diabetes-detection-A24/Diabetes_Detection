import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load your trained model
model_path = 'random_forest_model.joblib'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    st.error(f"Model file not found: {model_path}")

# Title of the app
st.title("Diabetes Detection App")

# Input fields for user data
st.write("Please enter the following details to predict if you have diabetes or not")

# Collect user input
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# Predict button
if st.button("Predict"):
    if os.path.exists(model_path):
        # Prepare the input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display the result
        if prediction[0] == 1:
            st.write("You are likely to have diabetes.")
        else:
            st.write("You are not likely to have diabetes.")
    else:
        st.error("Model file not found. Please upload the model file and try again.")