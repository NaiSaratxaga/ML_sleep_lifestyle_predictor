import streamlit as st

# Título
st.title("Mi primera app con Streamlit")

# Texto
st.write("¡Hola mundo! Esta es una app sencilla con Streamlit.")

# Input de texto
nombre = st.text_input("¿Cuál es tu nombre?")

# Botón
if st.button("Saludar"):
    st.write(f"Hola, {nombre} 👋")

# Gráfica de ejemplo
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'x': np.arange(10),
    'y': np.random.randn(10)
})

st.line_chart(data.set_index('x'))

# En la terminal ejecutar
# streamlit run app.py
