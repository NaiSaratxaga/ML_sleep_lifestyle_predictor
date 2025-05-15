# en la terminal ejecutar:
# streamlit run c:/Users/nsara/Desktop/naiara/Sleep_disorder_predictor_ML/src/app-streamlit/mi_app.py [ARGUMENTS]

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------------------------
# Cargar y preparar datos
# --------------------------
@st.cache_data
def cargar_modelo():
    df = pd.read_csv("src\data\sleep_predictor_dataset.csv")
    df = df.drop_duplicates()

    X = df.drop(columns=["Sleep_disorder", "Diagnosis_Confirmed"])
    y = df["Sleep_disorder"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

model, columnas = cargar_modelo()

# --------------------------
# Recomendaciones por tipo
# --------------------------
recomendaciones = {
   
    1: [
        "Establece un horario de sueño regular.",
        "Evita pantallas 1 hora antes de dormir.",
        "Practica meditación o respiración profunda."
    ],
    2: [
        "Consulta con un neurólogo.",
        "Evita manejar si tienes somnolencia.",
        "Haz siestas programadas si es posible."
    ],
    3: [
        "Perder peso si tienes sobrepeso.",
        "Evita alcohol y sedantes antes de dormir.",
        "Duerme de lado, no boca arriba."
    ],
    4: [
        "Evita cafeína por la tarde.",
        "Mantén una rutina de sueño relajante.",
        "Reduce el estrés antes de dormir."
    ],
    5: [
        "Consulta por niveles bajos de hierro.",
        "Haz masajes o estiramientos en las piernas.",
        "Mantén una temperatura fresca en el dormitorio."
    ]
}

# --------------------------
# Interfaz Streamlit
# --------------------------
st.title("🛌 Recomendador de Sueño Personalizado")

st.markdown("Introduce tus datos para predecir posibles trastornos del sueño y obtener recomendaciones.")

input_data = {}
input_data["Gender"] = st.selectbox("Género", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
input_data["Age"] = st.slider("Edad", 18, 80, 30)
input_data["Occupation"] = st.selectbox("Ocupación", range(0, 10))
input_data["Sleep Duration"] = st.slider("Duración del sueño (horas)", 0.0, 12.0, 6.0, step=0.5)
input_data["Quality of Sleep"] = st.slider("Calidad del sueño (1-10)", 1, 10, 5)
input_data["Physical Activity Level"] = st.slider("Nivel de actividad física (0-100)", 0, 100, 50)
input_data["Stress Level"] = st.slider("Nivel de estrés (0-10)", 0, 10, 5)
input_data["BMI Category"] = st.selectbox("Categoría de IMC", [0, 1, 2, 3])
input_data["Blood Pressure"] = st.slider("Presión arterial (valor promedio)", 70, 180, 120)
input_data["Heart Rate"] = st.slider("Frecuencia cardíaca", 40, 130, 75)
input_data["Daily Steps"] = st.slider("Pasos diarios", 0, 20000, 5000)

if st.button("Obtener Recomendación"):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]

    st.subheader(f"🔍 Trastorno del sueño predicho: {pred}")
    st.markdown("### 📝 Recomendaciones:")
    for r in recomendaciones.get(int(pred), ["Consulta con un especialista del sueño."]):
        st.markdown(f"- {r}")
