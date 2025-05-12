import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y vectorizador
model = joblib.load('modelo.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Definir la función limpiar_texto
def limpiar_texto(texto):
    # Ejemplo de limpieza de texto: convertir a minúsculas y eliminar espacios extra
    return " ".join(texto.lower().strip().split())

st.title("Recomendador de Trastornos del Sueño")

input_text = st.text_area("Texto OCR del paciente")

if st.button("Predecir"):
    texto_limpio = limpiar_texto(input_text)
    vect = vectorizer.transform([texto_limpio])
    pred = model.predict(vect)[0]
    # Definir la función recomendar
    def recomendar(pred):
        # Ejemplo de recomendaciones basadas en el diagnóstico
        recomendaciones = {
            "insomnio": "Evitar cafeína antes de dormir y establecer una rutina de sueño.",
            "apnea": "Consultar a un especialista en trastornos del sueño.",
            "narcolepsia": "Evitar actividades peligrosas y buscar tratamiento médico."
        }
        return recomendaciones.get(pred, "Consulta a un médico para más información.")

    rec = recomendar(pred)

    st.subheader(f"Diagnóstico: {pred}")
    st.markdown(f"**Recomendación:** {rec}")
