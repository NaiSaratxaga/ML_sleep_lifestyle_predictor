
# 💤 Dormir Mejor con Datos: Machine Learning para Detectar Trastornos del Sueño

Este proyecto tiene como finalidad analizar cómo los hábitos de vida y factores de salud influyen en la calidad del sueño. Utilizando un conjunto de datos obtenido desde [Kaggle](https://www.kaggle.com/datasets/ziya07/sleep-disorder-diagnostic-dataset/data), se recopila información sobre comportamientos diarios, condiciones médicas, niveles de estrés y patrones de sueño.

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

- **Patient_ID**	Identificador único de cada persona.
- **Age**	Edad del paciente (en años).
- **Gender**	Género de la persona (Male o Female).
- **Sleep_Disorder_Type** Diagnóstico del trastorno del sueño del paciente (categórico, con varias categorías como Apnea del Sueño, Insomnio, Narcolepsia, etc.).
- **AHI_Score** Puntaje del índice Apnea-Hipopnea del paciente (numérico).
- **SaO2_Level** Nivel de saturación de oxígeno en sangre del paciente (numérico).
- **OCR_Extracted_Text** Texto extraído mediante OCR de los documentos médicos escaneados (texto). **Necesita NPL para su análisis (Procesamiento de lenguaje natural)**
- **Diagnosis_Confirmed** Columna binaria que indica si el diagnóstico de trastorno del sueño está confirmado (1 para confirmado, 0 para no confirmado).

Tipos de trastornos
- Restless Leg Syndrome → Síndrome de Piernas Inquietas. Trastorno neurológico caracterizado por una necesidad incontrolable de mover las piernas, especialmente por la noche.
- Insomnia → Insomnio. Dificultad para conciliar el sueño, permanecer dormido o despertarse demasiado temprano y no poder volver a dormir.
- Narcolepsy → Narcolepsia. Trastorno del sueño que provoca somnolencia extrema durante el día y episodios súbitos de sueño.
- Obstructive Sleep Apnea → Apnea Obstructiva del Sueño. Trastorno en el que la respiración se interrumpe repetidamente durante el sueño debido a una obstrucción de las vías respiratorias.
- No Disorder → Sin Trastorno. Pacientes que no presentan ningún trastorno del sueño diagnosticado.


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
