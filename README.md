# 🚲 Modelado de Demanda de Bicicletas en Seúl

## 📋 Descripción del Proyecto

Este proyecto de Machine Learning tiene como objetivo predecir la demanda de bicicletas por hora en la ciudad de Seúl, Corea del Sur. Utilizando datos históricos del sistema de bicicletas compartidas y variables meteorológicas, se desarrollaron modelos predictivos para optimizar la gestión y disponibilidad de bicicletas en la ciudad.

## 📊 Dataset

El dataset **Seoul Bike Data** contiene información sobre el alquiler de bicicletas y condiciones climáticas en Seúl durante un año completo.

### Características del Dataset:
- **8,760 registros** (365 días × 24 horas)
- **14 variables**
- **Sin valores nulos**

### Variables:

| Variable | Descripción | Tipo |
|----------|-------------|------|
| Date | Fecha del registro | datetime |
| Rented Bike Count | Número de bicicletas alquiladas (Variable Objetivo) | int |
| Hour | Hora del día (0-23) | int |
| Temperature(°C) | Temperatura en grados Celsius | float |
| Humidity(%) | Porcentaje de humedad | int |
| Wind speed (m/s) | Velocidad del viento | float |
| Visibility (10m) | Visibilidad en metros | int |
| Dew point temperature(°C) | Temperatura del punto de rocío | float |
| Solar Radiation (MJ/m2) | Radiación solar | float |
| Rainfall(mm) | Precipitación en milímetros | float |
| Snowfall (cm) | Nevada en centímetros | float |
| Seasons | Estación del año (Winter, Spring, Summer, Autumn) | categorical |
| Holiday | Día festivo (Holiday, No Holiday) | categorical |
| Functioning Day | Día funcional del servicio | categorical |

## 🔬 Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Análisis de la distribución de la variable objetivo
- Identificación de patrones temporales
- Análisis de correlaciones entre variables
- Visualización de datos mediante gráficos interactivos

### 2. Preprocesamiento de Datos
- Ingeniería de características a partir de la fecha
- Codificación de variables categóricas (One-Hot Encoding)
- Escalado de variables numéricas (StandardScaler, MinMaxScaler)
- División de datos en conjuntos de entrenamiento y prueba

### 3. Modelado
Se evaluaron múltiples algoritmos de regresión:

| Modelo | Tipo |
|--------|------|
| Linear Regression | Regresión Lineal |
| Ridge Regression | Regresión con regularización L2 |
| Lasso Regression | Regresión con regularización L1 |
| ElasticNet | Combinación de L1 y L2 |
| Random Forest Regressor | Ensemble - Bagging |
| Gradient Boosting Regressor | Ensemble - Boosting |
| SVR (Support Vector Regression) | Kernel-based |
| KNeighbors Regressor | Instance-based |
| XGBoost Regressor | Ensemble - Boosting |
| LightGBM Regressor | Ensemble - Boosting |

### 4. Optimización de Hiperparámetros
- Búsqueda de hiperparámetros mediante RandomizedSearchCV
- Selección de características relevantes
- Validación cruzada para evitar overfitting

## 📈 Resultados

El modelo final **XGBoost Regressor** con hiperparámetros optimizados logró los mejores resultados:

| Métrica | Valor |
|---------|-------|
| RMSE (Test) | ~53.002 |
| RMSE (Train) | ~15.174 |

### Hiperparámetros del Modelo Final:
```python
XGBRegressor(
    random_state=42,
    subsample=0.6,
    reg_lambda=1,
    reg_alpha=0.01,
    n_estimators=1000,
    min_child_weight=5,
    max_depth=6,
    learning_rate=0.05,
    gamma=0.1,
    colsample_bytree=1.0
)
```

## 🛠️ Tecnologías Utilizadas

### Lenguaje
- Python 3.11

### Librerías Principales
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **scipy** - Análisis estadístico
- **matplotlib** - Visualización estática
- **seaborn** - Visualización estadística
- **plotly** - Visualización interactiva
- **scikit-learn** - Machine Learning
- **xgboost** - Gradient Boosting
- **lightgbm** - Gradient Boosting

## 📁 Estructura del Proyecto

```
Machine-Learning-Project-Portfolio/
│
├── Modelado_Demanda_bikes_Seul.ipynb    # Notebook principal
├── SeoulBikeData.csv                     # Dataset (requerido)
└── README.md                             # Este archivo
```

## 🚀 Cómo Ejecutar

1. Clonar el repositorio:
```bash
git clone https://github.com/samgarcia42/Machine-Learning-Project-Portfolio.git
```

2. Instalar las dependencias:
```bash
pip install pandas numpy scipy matplotlib seaborn plotly scikit-learn xgboost lightgbm
```

3. Descargar el dataset [Seoul Bike Data](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand) y colocarlo en el directorio del proyecto.

4. Ejecutar el notebook:
```bash
jupyter notebook Modelado_Demanda_bikes_Seul.ipynb
```

## 📝 Conclusiones

- Los modelos de ensemble (XGBoost, LightGBM) superan significativamente a los modelos lineales tradicionales.
- La optimización de hiperparámetros mejora notablemente el rendimiento del modelo.
- El modelo puede ser utilizado para optimizar la distribución de bicicletas según condiciones climáticas y temporales.

## 👤 Autor

**Sam Garcia**

---

⭐ Si este proyecto te fue útil, ¡no olvides darle una estrella!
