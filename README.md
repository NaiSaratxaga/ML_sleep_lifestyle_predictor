
# 💤 Dormir Mejor con Datos: Machine Learning para Detectar Trastornos del Sueño

Este proyecto tiene como finalidad analizar cómo los hábitos de vida y factores de salud influyen en la calidad del sueño. Utilizando un conjunto de datos obtenido desde [Kaggle](https://www.kaggle.com/), se recopila información sobre comportamientos diarios, condiciones médicas, niveles de estrés y patrones de sueño.

A través de técnicas de análisis exploratorio de datos (EDA) y algoritmos de aprendizaje automático, se busca identificar patrones significativos y construir un modelo predictivo capaz de detectar posibles trastornos del sueño y clasificar el tipo específico de trastorno.

## 🎯 Objetivo del Proyecto

- **Identificar los factores más influyentes para ayudar a mejorar los hábitos de sueño.**
- **Analizar los hábitos de sueño** según duración y calidad entre distintos grupos demográficos (edad, ocupación, género).
- **Explorar la relación entre el estilo de vida y la salud del sueño**, evaluando el impacto de factores como estrés, actividad física, IMC, consumo de alcohol o cafeína.
- **Predecir trastornos del sueño**, como insomnio, apnea del sueño o narcolepsia.
- **Clasificar la calidad del sueño** (bueno, regular, malo).
- **Generar recomendaciones personalizadas** para mejorar el descanso.

## 🧠 ¿Qué se Puede Predecir?

- **Tipo de predicción**: Clasificación binaria (trastorno del sueño: sí/no) seguida de una **clasificación multiclase** del tipo de trastorno (insomnio, apnea del sueño, narcolepsia, etc.).
- **Relevancia médica y social**: Anticipar estos trastornos permite una intervención temprana, mejora la calidad de vida y reduce riesgos en la salud pública.
- **Aplicabilidad práctica**: El modelo puede generar recomendaciones útiles, por ejemplo: “esta persona probablemente tiene insomnio”, lo cual puede asistir a médicos, clínicas o aplicaciones de autocuidado.

## 📂 Descripción del Dataset

Incluye las siguientes variables:

- Edad, Género, Ocupación  
- Duración y Calidad del Sueño, Nivel de Actividad Física  
- Nivel de Estrés, IMC, Presión Arterial, Frecuencia Cardíaca, Pasos Diarios  
- **Sleep_Disorder** (variable objetivo: sin trastorno, insomnio, apnea, narcolepsia)

## ⚙️ Herramientas y Librerías

- **Python**
- **Pandas**, **NumPy** para manipulación de datos  
- **Scikit-learn**, **XGBoost**, y **Random Forest** para modelos de clasificación  
- **Matplotlib** y **Seaborn** para visualizaciones  

## 🛠️ Metodología

1. **Preprocesamiento de Datos**  
   - Relleno de valores nulos  
   - Codificación de variables categóricas  
   - Escalado de características  
   - Análisis de correlación y selección de características

2. **División del Dataset**  
   - 80% entrenamiento, 20% prueba

3. **Modelos Utilizados**  
   - Regresión Logística  
   - KNN  
   - Random Forest  
   - XGBoost  
   - Redes Neuronales (para clasificación multiclase)

4. **Evaluación**  
   - Accuracy, Precision, Recall, F1-score  
   - Validación cruzada con k-fold  

## 👨‍💻 Autora

Proyecto realizado por **Naiara**.

## 🙏 Agradecimientos

Gracias a [Kaggle](https://www.kaggle.com/) por proporcionar el conjunto de datos utilizado en este proyecto.
