# mi_app.py

# mi_app.py
import streamlit as st

# ————————————————————————————————
# ¡SET_PAGE_CONFIG DEBE IR AQUÍ, ANTES DE CUALQUIER OTRO st.*
# ————————————————————————————————
st.set_page_config(
    page_title="Recomendador de Sueño",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import pickle
from pathlib import Path

# ——————————————————————————————
# 1) Carga de pipelines serializados
# ——————————————————————————————
@st.cache_resource
def load_models():
    root = Path(__file__).parent
    with open(root / "clasificacion_tipo_trastorno.pkl", "rb") as f:
        pipe_trastorno = pickle.load(f)
    with open(root / "modelo_diagnostico_confirmado.pkl", "rb") as f:
        pipe_confirmado = pickle.load(f)
    with open(root / "sleep_patient_segmentation.pkl", "rb") as f:
        pipe_segmentacion = pickle.load(f)
    return pipe_trastorno, pipe_confirmado, pipe_segmentacion

pipe_trastorno, pipe_confirmado, pipe_segmentacion = load_models()

# ——————————————————————————————
# 2) Mappings y recomendaciones
# ——————————————————————————————
class_labels = {
    0: "Insomnia", 1: "Narcolepsy", 2: "No Disorder",
    3: "Obstructive Sleep Apnea", 4: "Restless Leg Syndrome", 5: "Sleep Apnea"
}
recomendaciones = {
    # ... igual que antes ...
}

# ——————————————————————————————
# 3) Interfaz Streamlit
# ——————————————————————————————
st.title("🛌 Recomendador de Sueño Personalizado")
st.markdown("Rellena tus datos para predecir trastornos del sueño, diagnóstico confirmado y segmento de paciente.")

# … resto de tu código para inputs y predicciones …


# ——————————————————————————————
# 1) Carga de los pipelines serializados
# ——————————————————————————————
@st.cache_resource
def load_models():
    root = Path(__file__).parent
    with open(root / "clasificacion_tipo_trastorno.pkl", "rb") as f:
        pipe_trastorno = pickle.load(f)
    with open(root / "modelo_diagnostico_confirmado.pkl", "rb") as f:
        pipe_confirmado = pickle.load(f)
    with open(root / "sleep_patient_segmentation.pkl", "rb") as f:
        pipe_segmentacion = pickle.load(f)
    return pipe_trastorno, pipe_confirmado, pipe_segmentacion

pipe_trastorno, pipe_confirmado, pipe_segmentacion = load_models()

# ——————————————————————————————
# 2) Mappings y recomendaciones
# ——————————————————————————————
class_labels = {
    0: "Insomnia",
    1: "Narcolepsy",
    2: "No Disorder",
    3: "Obstructive Sleep Apnea",
    4: "Restless Leg Syndrome",
    5: "Sleep Apnea"
}

recomendaciones = {
    0: ["Establece un horario de sueño regular.",
        "Evita pantallas 1 h antes de dormir.",
        "Practica respiración profunda."],
    1: ["Consulta con un neurólogo.",
        "Evita manejar si tienes somnolencia.",
        "Haz siestas programadas."],
    2: ["Mantén tus hábitos saludables actuales."],
    3: ["Pierde peso si procede.",
        "Evita alcohol antes de dormir.",
        "Duerme de lado, no boca arriba."],
    4: ["Evita cafeína por la tarde.",
        "Rutina relajante antes de dormir.",
        "Reduce el estrés pre-sueño."],
    5: ["Consulta por niveles bajos de hierro.",
        "Estira las piernas antes de acostarte.",
        "Mantén el cuarto fresco."]
}

# ——————————————————————————————
# 3) Interfaz Streamlit
# ——————————————————————————————
st.set_page_config(page_title="Recomendador de Sueño", layout="wide")
st.title("🛌 Recomendador de Sueño Personalizado")

st.markdown("Rellena tus datos para predecir trastornos del sueño, diagnóstico confirmado y segmento de paciente.")

# Creamos un formulario para agrupar inputs
with st.form("form_inputs"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Género", [0,1], format_func=lambda x: "Femenino" if x==0 else "Masculino")
        age    = st.slider("Edad", 18, 80, 30)
        occ    = st.selectbox("Ocupación (código)", list(range(10)))
        sleep_dur = st.slider("Duración del sueño (h)", 0.0, 12.0, 6.0, 0.5)
        quality   = st.slider("Calidad del sueño (1-10)", 1,10,5)
    with col2:
        phys_act = st.slider("Actividad física (0-100)", 0,100,50)
        stress   = st.slider("Estrés (0-10)", 0,10,5)
        bmi_cat  = st.selectbox("Categoría IMC", [0,1,2,3])
        bp       = st.slider("Presión arterial", 70,180,120)
        hr       = st.slider("Frecuencia cardíaca", 40,130,75)
        steps    = st.slider("Pasos diarios", 0,20000,5000)

    submitted = st.form_submit_button("🔍 Predecir y recomendar")

if submitted:
    # Montar DataFrame de entrada
    X_new = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Occupation": occ,
        "Sleep Duration": sleep_dur,
        "Quality of Sleep": quality,
        "Physical Activity Level": phys_act,
        "Stress Level": stress,
        "BMI Category": bmi_cat,
        "Blood Pressure": bp,
        "Heart Rate": hr,
        "Daily Steps": steps
    }])

    # 1) Tipo de trastorno
    pred_t = pipe_trastorno.predict(X_new)[0]
    label_t = class_labels.get(int(pred_t), str(pred_t))

    # 2) Diagnóstico confirmado
    pred_c = pipe_confirmado.predict(X_new)[0]
    label_c = "✔️ Confirmado" if pred_c==1 else "❌ No confirmado"

    # 3) Segmentación de paciente
    pred_s = pipe_segmentacion.predict(X_new)[0]

    # Mostrar resultados
    st.subheader(f"1️⃣ Trastorno predicho: **{label_t}**")
    st.markdown("### Recomendaciones:")
    for r in recomendaciones.get(int(pred_t), ["Consulta con un especialista."]):
        st.markdown(f"- {r}")

    st.subheader(f"2️⃣ Diagnóstico confirmado: {label_c}")
    st.subheader(f"3️⃣ Segmento de paciente: Cluster **{pred_s}**")

