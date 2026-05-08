# 🚲 Predicción de la Demanda de Bicicletas Compartidas en Seúl

Proyecto de Machine Learning end-to-end cuyo objetivo es **predecir la demanda horaria de bicicletas compartidas en la ciudad de Seúl** a partir de variables meteorológicas, temporales y de calendario. Se comparan 10 algoritmos de regresión y se realiza un tuneo de hiperparámetros sobre el mejor candidato.

---

## 📋 Tabla de Contenidos

- [Descripción del proyecto](#-descripción-del-proyecto)
- [Dataset](#-dataset)
- [Estructura del notebook](#-estructura-del-notebook)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
- [Modelos evaluados](#-modelos-evaluados)
- [Tecnologías y librerías](#-tecnologías-y-librerías)
- [Instalación y uso](#-instalación-y-uso)
- [Resultados](#-resultados)
- [Aprendizajes clave](#-aprendizajes-clave)
- [Autor](#-autor)

---

## 📖 Descripción del proyecto

Las ciudades cada vez apuestan más por sistemas de bicicletas compartidas como alternativa de movilidad sostenible. Para que el servicio funcione bien es fundamental **anticipar la demanda hora a hora** y así garantizar disponibilidad de bicicletas en cada estación.

Este proyecto recorre todas las etapas típicas de un proyecto de Data Science:

1. Análisis Exploratorio de Datos (EDA)
2. Feature Engineering y preprocesado
3. Modelado y comparación de algoritmos
4. Selección de variables
5. Tuneo de hiperparámetros
6. Persistencia del modelo final con `pickle`

---

## 📊 Dataset

- **Fuente:** [UCI Machine Learning Repository - Seoul Bike Sharing Demand](https://archive-beta.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
- **Variable objetivo:** `Rented Bike Count` (número de bicicletas alquiladas por hora)
- **Variables predictoras:** información meteorológica (temperatura, humedad, viento, visibilidad, radiación solar, lluvia, nieve, punto de rocío), temporal (fecha, hora) y de calendario (estación del año, día festivo, día funcional).

> 📥 Para ejecutar el notebook, descarga el archivo `SeoulBikeData.csv` desde el enlace anterior y colócalo en el mismo directorio que el notebook.

---
---

## ⚙️ Pipeline de Machine Learning

### 🔍 Análisis Exploratorio
- Estudio de la distribución de la variable objetivo (asimetría positiva, Q-Q plot).
- Análisis de cardinalidad y frecuencia relativa de variables categóricas.
- Detección visual de outliers con boxplots e histogramas.
- Análisis de correlación entre features y la variable objetivo.
- Detección de **multicolinealidad** entre `Temperature` y `Dew point temperature`.

### 🛠️ Feature Engineering
A partir de la información temporal y meteorológica se construyen variables que ayudan a los modelos a capturar la estacionalidad y los patrones horarios:

- **Temporales:** `DayOfWeek`, `Month`, `Year`, `WeekStatus` (entresemana / fin de semana), `hora_bin` (Madrugada, Mañana, Tarde, Noche).
- **Hora pico:** flag binario para horarios de entrada/salida laboral (8h y 18h).
- **Estadísticos de demanda:** demanda promedio y mediana agrupada por hora, día de la semana, y combinación día-hora.
- **Variables rezagadas (lags):** valores de hace 1h, 24h y 168h (1 semana) para target y variables meteorológicas.
- **Medias móviles:** ventanas de 3h, 6h, 12h y 24h.

### 🧪 Validación
Se utiliza un **split temporal** (no aleatorio) para respetar la naturaleza secuencial de los datos y evitar *data leakage*:
- **Train:** desde el inicio hasta el 31/10/2018.
- **Test:** desde el 01/11/2018 en adelante.

---

## 🤖 Modelos evaluados

Se entrenaron y compararon 10 algoritmos diferentes:

| Familia | Modelos |
|---|---|
| **Lineales** | Linear Regression (baseline), Ridge, Lasso, ElasticNet |
| **Basados en distancia** | K-Nearest Neighbors |
| **Ensembles - Bagging** | Random Forest |
| **Ensembles - Boosting** | Gradient Boosting, XGBoost, LightGBM |
| **Kernel** | Support Vector Regressor (SVR) |

**Métricas de evaluación:** `RMSE` y `MAE`, calculadas tanto en train como en test para detectar overfitting.

Los modelos basados en árboles (Random Forest, XGBoost, LightGBM) ofrecieron el mejor rendimiento, lo cual era esperable dada la naturaleza no lineal de la relación entre las features y la variable objetivo.

---

## 🎯 Tuneo de hiperparámetros

Sobre el mejor modelo (**XGBoost**) se aplica `RandomizedSearchCV` con validación cruzada de 5 folds, explorando combinaciones de:

- `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`
- `gamma`, `subsample`, `colsample_bytree`
- `reg_alpha`, `reg_lambda`

Posteriormente se hace **selección de variables** filtrando por importancia (umbral ≥ 0.0015) y se reentrena el modelo final, que se serializa con `pickle` para su posterior despliegue.

---

## 🛠️ Tecnologías y librerías

```python
# Manipulación y análisis
pandas, numpy, scipy

# Visualización
matplotlib, seaborn, plotly

# Preprocesado
scikit-learn (StandardScaler, OneHotEncoder)

# Modelado
scikit-learn (LinearRegression, Ridge, Lasso, ElasticNet,
              RandomForestRegressor, GradientBoostingRegressor,
              SVR, KNeighborsRegressor)
xgboost (XGBRegressor)
lightgbm (LGBMRegressor)

# Optimización y persistencia
scikit-learn (RandomizedSearchCV)
pickle
```

---

## 📈 Resultados

El modelo final corresponde a un **XGBoost Regressor tuneado** con selección previa de variables, capaz de capturar tanto los patrones horarios y semanales de la demanda como el efecto de las condiciones meteorológicas.

Las predicciones se comparan visualmente contra los valores reales del conjunto de test mediante gráficos interactivos de Plotly, lo cual permite observar el ajuste del modelo a lo largo del tiempo.

---

## 💡 Aprendizajes clave

Este proyecto puede servir como referencia para aprender o repasar conceptos como:

- ✅ Análisis exploratorio sistemático con funciones reutilizables.
- ✅ Feature engineering temporal (lags, rolling windows, binning de horas).
- ✅ Importancia del **split temporal** en problemas de series de tiempo.
- ✅ Comparación rigurosa de múltiples modelos bajo las mismas métricas.
- ✅ Detección y manejo de multicolinealidad.
- ✅ Tuneo eficiente de hiperparámetros con `RandomizedSearchCV` vs `GridSearchCV`.
- ✅ Selección de variables basada en feature importance.
- ✅ Persistencia de modelos con `pickle` para producción.

---

## 🤝 Contribuciones

Este notebook forma parte de un portafolio de proyectos de Machine Learning con fines didácticos. Si tienes sugerencias, encuentras un error o quieres compartir mejoras, ¡las contribuciones son bienvenidas! Puedes abrir un *issue* o un *pull request*.

---

## 👤 Autor

Proyecto desarrollado como parte de un portafolio personal de Machine Learning.

Si te resultó útil, ⭐ no olvides darle una estrella al repositorio.

---

> _"En Dios confiamos. Todos los demás deben traer datos."_ — W. Edwards Deming
