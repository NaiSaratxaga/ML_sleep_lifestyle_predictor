

# Diseño o App sencilla para mostrar resultados
# App web con Streamlit 

import streamlit as st
import joblib
import numpy as np

# Cargar modelo entrenado
model = joblib.load('C:/Users/nsara/Desktop/naiara_thebridge/Sleep_disorder_predictor_ML/models/sleep_model.pkl')


# Título de la app
st.title("🛌 Predicción de Sueño")

st.write("Completa los campos para predecir tu calidad de sueño o las horas recomendadas.")

# Entradas del usuario
edad = st.number_input("Edad", min_value=0, max_value=100, value=25)
horas_pantalla = st.slider("Horas de pantalla por día", 0.0, 16.0, step=0.5, value=6.0)
cafe = st.selectbox("Tazas de café al día", [0, 1, 2, 3, 4, 5])
estres = st.slider("Nivel de estrés (1=calmado, 10=muy estresado)", 1, 10, value=5)
ejercicio = st.selectbox("¿Haces ejercicio regularmente?", ["Sí", "No"])

# Convertir entrada en formato numérico si es necesario
ejercicio_bin = 1 if ejercicio == "Sí" else 0

# Botón para predecir
if st.button("Predecir sueño"):
    # Crear vector de entrada para el modelo
    X_input = np.array([[edad, horas_pantalla, cafe, estres, ejercicio_bin]])

    # Predecir
    pred = model.predict(X_input)[0]

    # Mostrar resultado
    st.success(f"Predicción del modelo: {pred}")
    
    

