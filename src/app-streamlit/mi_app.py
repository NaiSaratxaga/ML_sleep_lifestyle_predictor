# src/app-streamlit/mi_app.py

import os
import pickle
import streamlit as st
import pandas as pd

# 1) Esto **debe** ir primero en tu script
st.set_page_config(page_title="Recomendador de Sueño", layout="wide")

# 2) Importa la clase para que pickle la encuentre
from threshold_classifier import ThresholdClassifier

# 3) Carga tus pipelines
@st.cache_data
def load_pipelines():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "clasificacion_tipo_trastorno.pkl"),   "rb") as f:
        pipe_tipo = pickle.load(f)
    with open(os.path.join(base, "modelo_diagnostico_confirmado.pkl"), "rb") as f:
        pipe_diag = pickle.load(f)
    with open(os.path.join(base, "sleep_patient_segmentation.pkl"),   "rb") as f:
        pipe_seg  = pickle.load(f)
    return pipe_tipo, pipe_diag, pipe_seg

pipe_tipo, pipe_diag, pipe_seg = load_pipelines()

# 4) … el resto de tu app (inputs, botón “Analizar”, predicciones, etc.) …
