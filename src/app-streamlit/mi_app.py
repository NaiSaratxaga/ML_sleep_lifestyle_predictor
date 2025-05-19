# mi_app.py

import streamlit as st
import pandas as pd
import pickle

# 1) ¡Debe ser la PRIMERA llamada a Streamlit!
st.set_page_config(page_title="Recomendador de Sueño", layout="wide")

# 2) Carga los tres pipelines desde disco
@st.cache_data
def load_pipelines():
    with open("clasificacion_tipo_trastorno.pkl",   "rb") as f:
        pipe_tipo = pickle.load(f)
    with open("modelo_diagnostico_confirmado.pkl", "rb") as f:
        pipe_diag = pickle.load(f)
    with open("sleep_patient_segmentation.pkl",   "rb") as f:
        pipe_seg  = pickle.load(f)
    return pipe_tipo, pipe_diag, pipe_seg

pipe_tipo, pipe_diag, pipe_seg = load_pipelines()

# 3) Diccionario de recomendaciones según tipo de trastorno
recomendaciones = {
    0: ["Consulta con un especialista del sueño."],
    1: ["Establece un horario de sueño regular.",
        "Evita pantallas 1 h antes de dormir.",
        "Practica meditación o respiración profunda."],
    2: ["Consulta con un neurólogo.",
        "Evita manejar con somnolencia.",
        "Haz siestas programadas si es posible."],
    3: ["Perder peso si tienes sobrepeso.",
        "Evita alcohol y sedantes antes de dormir.",
        "Duerme de lado, no boca arriba."],
    4: ["Evita cafeína por la tarde.",
        "Mantén una rutina de sueño relajante.",
        "Reduce el estrés antes de dormir."],
    5: ["Consulta por niveles bajos de hierro.",
        "Haz masajes o estiramientos en las piernas.",
        "Mantén una temperatura fresca en el dormitorio."]
}

# 4) Interfaz
st.title("🛌 Recomendador de Sueño Personalizado")
st.markdown("Rellena tus datos para predecir tipo de trastorno, diagnóstico confirmado y segmento de paciente.")

# Inputs
input_data = {}
input_data["Gender"]                   = st.selectbox("Género",                    [0,1], format_func=lambda x: "Femenino" if x==0 else "Masculino")
input_data["Age"]                      = st.slider("Edad",                       18,80,30)
input_data["Occupation"]               = st.selectbox("Ocupación (código)",        list(range(10)))
input_data["Sleep Duration"]           = st.slider("Duración del sueño (h)",     0.0,12.0,6.0,step=0.5)
input_data["Quality of Sleep"]         = st.slider("Calidad del sueño (1–10)",   1,10,5)
input_data["Physical Activity Level"]  = st.slider("Actividad física (0–100)",   0,100,50)
input_data["Stress Level"]             = st.slider("Nivel de estrés (0–10)",      0,10,5)
input_data["BMI Category"]             = st.selectbox("Categoría de IMC",          [0,1,2,3])
input_data["Blood Pressure"]           = st.slider("Presión arterial",           70,180,120)
input_data["Heart Rate"]               = st.slider("Frecuencia cardíaca",       40,130,75)
input_data["Daily Steps"]              = st.slider("Pasos diarios",             0,20000,5000)

# 5) Al pulsar “Analizar”
if st.button("Analizar"):
    input_df = pd.DataFrame([input_data])

    # 6) Predicciones
    tipo_pred = pipe_tipo.predict(input_df)[0]
    diag_pred = pipe_diag.predict(input_df)[0]
    seg_pred  = pipe_seg.predict(input_df)[0]

    # 7) Mostrar resultados
    st.subheader("🔍 Resultados de la Predicción")
    st.write(f"• Tipo de trastorno: **{tipo_pred}**")
    st.write(f"• Diagnóstico confirmado: **{diag_pred}**")
    st.write(f"• Segmento de paciente: **{seg_pred}**")

    # 8) Recomendaciones
    st.markdown("### 📝 Recomendaciones:")
    for r in recomendaciones.get(tipo_pred, recomendaciones[0]):
        st.markdown(f"- {r}")
