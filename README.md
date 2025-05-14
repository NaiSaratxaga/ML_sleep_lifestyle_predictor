
# 💤 Predicción de Trastornos del Sueño con Machine Learning

Este proyecto utiliza algoritmos de aprendizaje automático para predecir trastornos del sueño a partir de datos relacionados con la salud y el estilo de vida, con el objetivo de facilitar un diagnóstico temprano y cuidados preventivos.

## 📂 Descripción General

El propósito de este proyecto es analizar los factores que afectan la salud del sueño y predecir qué tipo de trastorno del sueño puede tener una persona, basándose en variables como:

- Edad, Género, Ocupación  
- Duración y Calidad del Sueño, Nivel de Actividad Física  
- Nivel de Estrés, IMC (Índice de Masa Corporal), Presión Arterial, Frecuencia Cardíaca, Pasos Diarios  
- Trastorno del Sueño (variable objetivo: 1 si tiene trastorno, 0 si no)

El conjunto de datos fue obtenido desde [Kaggle](https://www.kaggle.com/).

## ⚙️ Herramientas y Librerías

El proyecto fue desarrollado en Python, utilizando:

- **Pandas** y **NumPy** para manipulación de datos  
- **Scikit-learn** para modelado y entrenamiento  
- **Matplotlib** y **Seaborn** para visualización  

## 🛠️ Metodología

1. **Preprocesamiento de Datos**:  
   - Relleno de valores nulos  
   - Codificación de variables categóricas  
   - Escalado de características  
   - Análisis de importancia de variables

2. **División del Conjunto de Datos**:  
   - División en 75% entrenamiento y 25% prueba

3. **Modelos Utilizados**:
   - Regresión Logística  
   - Clasificador K-Vecinos Más Cercanos (KNN)  
   - Clasificador Random Forest

4. **Evaluación**:
   - Se utilizaron métricas como **Precisión (Accuracy)**, **Recall**, **Precisión (Precision)** y **F1-score**  
   - Validación cruzada con *k-fold*

## 📊 Observaciones Clave (EDA)

- Personas mayores de 43 años son más propensas a sufrir trastornos del sueño  
- Las mujeres tienden a tener mejor calidad de sueño que los hombres  
- Ingenieros duermen mejor, mientras que los vendedores presentan peor calidad de sueño  
- Niveles altos de estrés están fuertemente relacionados con trastornos del sueño  
- Las personas con trastornos del sueño tienen calificaciones más bajas en calidad de sueño  
- Aquellos con IMC en categoría de sobrepeso u obesidad tienden a tener más trastornos  
- Dormir más de 7 horas disminuye significativamente la probabilidad de tener un trastorno  

## 📈 Resultados

- **Mejor Modelo**: Random Forest  
  - Precisión (Accuracy): **89%**  
  - Recall: 89%  
  - Precisión: 90%  
  - F1-score: 89%

- **Otros Modelos**:  
  - KNN: desempeño aceptable, pero inferior al Random Forest  
  - Regresión Logística: 86% de precisión, el más bajo de los tres  

### 🔍 Características Más Importantes

Según el modelo Random Forest, las tres características más relevantes para predecir trastornos del sueño son:

- Presión Arterial  
- Categoría de IMC  
- Edad  

## 📚 Inspiración

Este proyecto fue inspirado en el artículo de IEEE:  
*"Applying Machine Learning Algorithms for Classification of Sleep Disorder"*  

## 👨‍💻 Autor

Proyecto realizado por **Hrithik Manda**.

## 🙏 Agradecimientos

Gracias especiales a [Kaggle](https://www.kaggle.com/) por proporcionar el conjunto de datos utilizado en este proyecto.
