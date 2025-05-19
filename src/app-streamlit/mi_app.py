# mi_app.py
# cd src/app-streamlit
# streamlit run mi_app.py

import os
import streamlit as st
import pandas as pd
import pickle

# 1) Configuración de la página —¡DEBE SER LO PRIMERO!
st.set_page_config(page_title="Recomendador de Sueño", layout="wide")

# 2) Sidebar: logo + menú
BASE_DIR = os.path.dirname(__file__)
logo_path = os.path.join(BASE_DIR, "header.png")

st.sidebar.image(logo_path, width=120)
page = st.sidebar.selectbox("Menú", ["Inicio", "Ayuda"])

if page == "Ayuda":
    st.sidebar.markdown(
        """
        **¿Cómo usar la app?**
        1. Ve a "Inicio".  
        2. Introduce tu nombre e datos de sueño.  
        3. Pulsa "Analizar" y obtén tus recomendaciones.
        """
    )
    st.sidebar.markdown("---")

# 3) Carga los pipelines
@st.cache_data
def load_pipelines():
    with open(os.path.join(BASE_DIR, "clasificacion_tipo_trastorno.pkl"),   "rb") as f:
        pipe_tipo = pickle.load(f)
    with open(os.path.join(BASE_DIR, "modelo_diagnostico_confirmado.pkl"), "rb") as f:
        pipe_diag = pickle.load(f)
    with open(os.path.join(BASE_DIR, "sleep_patient_segmentation.pkl"),     "rb") as f:
        pipe_seg  = pickle.load(f)
    return pipe_tipo, pipe_diag, pipe_seg

pipe_tipo, pipe_diag, pipe_seg = load_pipelines()

# 4) Lista de columnas originales (solo para mostrar, no se usa directamente)
FEATURE_NAMES = [
    "Gender",
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps",
    "Occupation",
    "BMI Category"
]

if page == "Inicio":
    # 5) Título y descripción
    st.title("🛌 Recomendador de Sueño Personalizado")
    st.markdown("🔍 **Introduce tus datos y descubramos juntos qué tipo de trastorno de sueño podrías estar padeciendo.**")

    # 6) Pedir nombre
    nombre = st.text_input("¿Cómo te llamas?", "")
    if nombre:
        st.success(f"¡Gracias, {nombre}! Ahora introduce tus datos:")

    # 7) Inputs principales
    input_data = {
        "Gender":                  st.selectbox("Género",                    [0,1], format_func=lambda x: "Femenino" if x==0 else "Masculino"),
        "Age":                     st.slider("Edad",                       18,80,30),
        "Sleep Duration":          st.slider("Duración del sueño (h)",     0.0,12.0,6.0,step=0.5),
        "Quality of Sleep":        st.slider("Calidad del sueño (1-10)",  1,10,5),
        "Physical Activity Level": st.slider("Actividad física (0-100)",   0,100,50),
        "Stress Level":            st.slider("Nivel de estrés (0-10)",      0,10,5),
        "Blood Pressure":          st.slider("Presión arterial",          70,180,120),
        "Heart Rate":              st.slider("Frecuencia cardíaca",       40,130,75),
        "Daily Steps":             st.slider("Pasos diarios",             0,20000,5000),
        "Occupation":              st.selectbox("Ocupación", ["Engineer", "Doctor", "Nurse", "Lawyer", "Teacher"]),
        "BMI Category":            st.selectbox("Categoría IMC", ["Normal", "Overweight", "Obese", "Underweight"])
    }

    # 8) Botón de análisis en el sidebar
    if st.sidebar.button("Analizar"):
        # 8.1) Crear DataFrame base
        df_in = pd.DataFrame([input_data])

        # 8.2) One-hot encoding de variables categóricas
        df_in = pd.get_dummies(df_in)

        # 8.3) Obtener columnas esperadas del modelo entrenado
        try:
            expected_cols = pipe_seg.feature_names_in_
        except AttributeError:
            st.error("El modelo no contiene 'feature_names_in_'. ¿Lo entrenaste con un DataFrame y sklearn >=1.0?")
            st.stop()

        # 8.4) Añadir columnas faltantes con valor 0
        for col in expected_cols:
            if col not in df_in.columns:
                df_in[col] = 0

        # 8.5) Asegurar el orden correcto
        df_in = df_in[list(expected_cols)]

        # 8.6) Ejecutar predicciones
        tipo_code = pipe_tipo.predict(df_in)[0]
        diag_pred = pipe_diag.predict(df_in)[0]
        seg_pred  = pipe_seg.predict(df_in)[0]

        # 9) Mapeo de códigos a nombres
        disorder_mapping = {
            0: "Insomnio",
            1: "Narcolepsia",
            2: "Sin trastorno",
            3: "Apnea (OSA)",
            4: "Síndrome de piernas inquietas",
            5: "Apnea del sueño"
        }
        tipo_name = disorder_mapping.get(tipo_code, f"Código desconocido ({tipo_code})")

        # 10) Mostrar resultados
        st.subheader("🔍 Resultados")
        st.write(f"• Tipo de trastorno: **{tipo_name}**")
        st.write(f"• Diagnóstico confirmado: **{diag_pred}**")
        st.write(f"• Segmento de paciente: **{seg_pred}**")

        # 11) Recomendaciones
        recomendaciones = {
            "Insomnio": [
                "Establece un horario de sueño fijo.",
                "Evita pantallas 1 h antes de dormir.",
                "Practica meditación o respiración profunda."
            ],
            "Narcolepsia": [
                "Consulta con un neurólogo.",
                "Evita manejar con somnolencia.",
                "Realiza siestas programadas."
            ],
            "Sin trastorno": [
                "¡Fenomenal! Sigue manteniendo tus hábitos de sueño."
            ],
            "Apnea (OSA)": [
                "Duerme de lado, no boca arriba.",
                "Evita alcohol y sedantes antes de dormir.",
                "Consulta con un especialista del sueño."
            ],
            "Síndrome de piernas inquietas": [
                "Consulta por niveles bajos de hierro.",
                "Masajes o estiramientos en piernas.",
                "Mantén ambiente fresco en el dormitorio."
            ],
            "Apnea del sueño": [
                "Consulta con un especialista del sueño.",
                "Evita alcohol y sedantes antes de dormir.",
                "Realiza chequeos periódicos de oxigenación."
            ]
        }

        st.markdown("### 📝 Recomendaciones:")
        for r in recomendaciones.get(tipo_name, recomendaciones["Sin trastorno"]):
            st.markdown(f"- {r}")
