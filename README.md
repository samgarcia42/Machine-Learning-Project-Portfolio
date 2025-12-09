# Machine Learning Project Portfolio

## 📊 Acerca de este Repositorio

Bienvenido a mi portafolio de proyectos de Machine Learning y Data Science. Este repositorio tiene como objetivo compartir mis conocimientos y habilidades en el campo del aprendizaje automático y la ciencia de datos, con la intención de colaborar y aprender junto a otras personas del campo.

Aquí encontrarás proyectos completos que demuestran diferentes técnicas, algoritmos y metodologías utilizadas en Machine Learning, desde el análisis exploratorio de datos hasta el desarrollo y optimización de modelos predictivos.

---

## 🚴 Proyecto 1: Predicción de la Demanda de Bicicletas Compartidas en Seúl

### Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar y comparar distintos algoritmos de machine learning para predecir la demanda de bicicletas compartidas en Seúl, Corea del Sur. El análisis completo sigue las etapas fundamentales de un proyecto de Machine Learning y Data Science profesional.

### 🎯 Objetivos

- Realizar un análisis exploratorio exhaustivo de los datos (EDA)
- Implementar técnicas de preprocesamiento y feature engineering
- Entrenar y evaluar múltiples modelos de machine learning
- Comparar el rendimiento de diferentes algoritmos
- Optimizar el mejor modelo mediante tuneo de hiperparámetros

### 📁 Dataset

Los datos fueron extraídos del **UCI Machine Learning Repository** y contienen información sobre el sistema de bicicletas compartidas de Seúl.

- **Fuente de datos**: [Seoul Bike Sharing Demand Dataset](https://archive-beta.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
- **Características del dataset**:
  - 8,760 registros (datos por hora durante un año completo)
  - 14 columnas incluyendo variables meteorológicas, temporales y la demanda de bicicletas
  - Variables categóricas: Estaciones, condiciones climáticas, días festivos
  - Variables numéricas: Temperatura, humedad, velocidad del viento, visibilidad, etc.

### 🔧 Metodología

El proyecto implementa las siguientes etapas:

1. **Análisis Exploratorio de Datos (EDA)**
   - Análisis de la distribución de la variable objetivo
   - Exploración de relaciones entre variables categóricas y la demanda
   - Análisis de correlaciones entre variables numéricas
   - Detección de valores atípicos (outliers)

2. **Preprocesamiento de Datos**
   - Tratamiento de valores faltantes
   - Normalización y estandarización de variables numéricas
   - Codificación de variables categóricas (One-Hot Encoding)
   - Feature engineering

3. **Modelado y Evaluación**
   - Implementación de múltiples algoritmos de regresión
   - Evaluación mediante métricas como MSE y MAE
   - Comparación de rendimiento entre modelos

4. **Optimización**
   - Tuneo de hiperparámetros del mejor modelo
   - Validación cruzada
   - Análisis de resultados finales

### 🤖 Algoritmos Implementados

El proyecto incluye la implementación y comparación de los siguientes algoritmos:

- **Modelos Lineales**: Linear Regression, Ridge, Lasso, ElasticNet
- **Modelos Basados en Árboles**: Random Forest, Gradient Boosting
- **Modelos de Boosting Avanzados**: XGBoost, LightGBM
- **Otros Modelos**: Support Vector Regression (SVR), K-Nearest Neighbors

### 📚 Tecnologías y Librerías Utilizadas

**Análisis y Manipulación de Datos:**
- pandas
- numpy
- scipy

**Visualización:**
- matplotlib
- seaborn
- plotly

**Machine Learning:**
- scikit-learn
- xgboost
- lightgbm

### 📈 Resultados

El notebook incluye análisis detallados de:
- Distribución de la demanda de bicicletas (con asimetría positiva)
- Patrones temporales y estacionales
- Impacto de variables meteorológicas
- Comparación de métricas de rendimiento entre modelos
- Modelo óptimo con hiperparámetros ajustados

### 🔍 Cómo Utilizar este Proyecto

1. **Descargar el dataset** desde el enlace proporcionado del UCI Machine Learning Repository
2. **Abrir el notebook**: `Modelado_Demanda_bikes_Seul.ipynb`
3. **Ejecutar las celdas** secuencialmente para reproducir el análisis completo

---

## 🤝 Colaboración

Este repositorio está abierto a colaboraciones, sugerencias y mejoras. Si tienes ideas o comentarios sobre los proyectos, no dudes en:
- Abrir un issue
- Proponer mejoras
- Compartir tu experiencia

## 📬 Contacto

Si estás interesado en colaborar o discutir sobre Machine Learning y Data Science, ¡no dudes en contactarme!

---

## 📄 Licencia

Este proyecto está disponible como código abierto para fines educativos y de aprendizaje.

---

*Este portafolio está en constante evolución. Pronto se agregarán más proyectos de Machine Learning y Data Science.*
