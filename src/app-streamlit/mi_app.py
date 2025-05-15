# en la terminal ejecutar:
# streamlit run c:/Users/nsara/Desktop/naiara/Sleep_disorder_predictor_ML/src/app-streamlit/mi_app.py [ARGUMENTS]


import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
 
st.title('Customer Churn Prediction')
 
# Load your trained model
model = RandomForestClassifier()
model.load('my_model.pkl')
 
# Get user input
age = st.slider('Age', 18, 100)
tenure = st.slider('Tenure (months)', 1, 72)
monthly_charges = st.slider('Monthly Charges', 20.0, 100.0)
 
# Make prediction
input_data = np.array([[age, tenure, monthly_charges]])
prediction = model.predict_proba(input_data)
 
# Display prediction
st.write(f'Churn Probability: {prediction[0][1]}')