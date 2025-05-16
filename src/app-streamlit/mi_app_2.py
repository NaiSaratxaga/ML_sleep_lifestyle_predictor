# -*- coding: utf-8 -*-
# streamlit run c:/Users/nsara/Desktop/naiara/Sleep_disorder_predictor_ML/src/app-streamlit/mi_app_2.py [ARGUMENTS
"""
Streamlit app para clasificación de trastorno de sueño
Usa el dataset combined_sleep_dataset.csv y RandomForestClassifier.
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# —————————————————————————————————————————————————————————————
# 0. Configuración de la página (debe ir al inicio)
# —————————————————————————————————————————————————————————————
st.set_page_config(
    page_title="Sleep Disorder Classifier",
    layout="centered"
)

# Mapeo de códigos a nombres de trastorno
SLEEP_NAMES = {
    0: 'Insomnia',
    1: 'Narcolepsy',
    2: 'No Disorder',
    3: 'Obstructive Sleep Apnea',
    4: 'Restless Leg Syndrome',
    5: 'Sleep Apnea'
}

# —————————————————————————————————————————————————————————————
# 1. Carga y split de datos (cacheados)
# —————————————————————————————————————————————————————————————
@st.cache_data
def load_and_split_data(path='combined_sleep_dataset.csv'):
    df = pd.read_csv(path)

    numeric_feats = [
        'Age',
        'Sleep Duration',
        'Quality of Sleep',
        'Physical Activity Level',
        'Stress Level',
        'Heart Rate',
        'Daily Steps',
        'Diagnosis_Confirmed'
    ]
    cate_feats = ['BMI Category', 'Blood Pressure']

    features = df[numeric_feats + cate_feats]
    target = df['Sleep_disorder']

    x_train, x_test, y_train, y_test = train_test_split(
        features, target,
        test_size=0.2,
        stratify=target,
        random_state=42
    )
    return features, x_train, x_test, y_train, y_test

# —————————————————————————————————————————————————————————————
# 2. Entrenamiento del modelo
# —————————————————————————————————————————————————————————————
@st.cache_resource
def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model

# —————————————————————————————————————————————————————————————
# 3. Sidebar para inputs del usuario
# —————————————————————————————————————————————————————————————
def construct_sidebar(features):
    st.sidebar.markdown("## 👤 Parámetros del paciente")

    inputs = []
    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]):
            min_val = float(features[col].min())
            max_val = float(features[col].max())
            med_val = float(features[col].median())
            val = st.sidebar.slider(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=med_val
            )
        else:
            options = sorted(features[col].unique())
            val = st.sidebar.selectbox(label=col, options=options)
        inputs.append(val)

    return inputs

# —————————————————————————————————————————————————————————————
# 4. Gráfico de pastel de probabilidades
# —————————————————————————————————————————————————————————————
def plot_pie_chart(probabilities):
    fig = go.Figure(
        data=[go.Pie(
            labels=list(SLEEP_NAMES.values()),
            values=probabilities[0]
        )]
    )
    fig.update_traces(
        hoverinfo='label+percent',
        textinfo='value',
        textfont_size=14
    )
    return fig

# —————————————————————————————————————————————————————————————
# 5. Mostrar resultados
# —————————————————————————————————————————————————————————————
def display_results(prediction, probabilities):
    pred_code = int(prediction[0])
    pred_name = SLEEP_NAMES[pred_code]

    st.markdown("## 🛌 Predicción de trastorno del sueño")
    col1, col2 = st.columns(2)

    col1.markdown("**Trastorno predicho**")
    col1.write(f"### {pred_name}")

    col2.markdown("**Confianza (probabilidad)**")
    col2.write(f"### {probabilities[0][pred_code]:.2f}")

    st.markdown("**Distribución de probabilidades**")
    fig = plot_pie_chart(probabilities)
    st.plotly_chart(fig, use_container_width=True)

# —————————————————————————————————————————————————————————————
# 6. Función principal
# —————————————————————————————————————————————————————————————
def main():
    st.title("🔍 Clasificación de trastorno de sueño")

    # Carga y split
    features, x_train, x_test, y_train, y_test = load_and_split_data()

    # Entrenamiento
    model = train_model(x_train, y_train)

    # Sidebar y predicción
    user_inputs = construct_sidebar(features)
    X_new = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    # Resultados
    display_results(prediction, probabilities)

if __name__ == "__main__":
    main()
