# Modelado de Riesgo Crediticio — Scorecard + Modelos Ensemble

## Descripción General

Este proyecto resuelve un problema clásico de **riesgo crediticio**: estimar la **probabilidad de impago (PD)** de un solicitante de un préstamo personal y traducir esa probabilidad a una decisión de negocio (aprobar / rechazar) y a un **precio (tasa de interés)** acorde al riesgo.

El objetivo es construir un sistema interpretable, auditable y rentable, alineado con la práctica bancaria tradicional (*scorecard*) pero comparado contra modelos *machine learning* modernos (XGBoost, LightGBM) para validar que la simplicidad no sacrifica poder predictivo. La salida es un **score de 100 a 1000 puntos** (mayor score → menor riesgo) junto con una propuesta de **pricing por tiers** (bandas A, B, C…).

El proyecto cubre el ciclo completo: definición del target con criterio anti-leakage, EDA, ingeniería de variables (WOE/IV), modelado, elección de punto de corte cost-sensitive, calibración isotónica de probabilidades, propuesta de pricing y empaquetado en un pipeline reproducible serializado con `joblib`.

> **Versión renderizada:** [`Challenge_Riesgo_Crediticio.html`](./Challenge_Riesgo_Crediticio.html) — notebook ejecutado de extremo a extremo, con todas las gráficas y outputs visibles directamente en el navegador.

## Estructura de Datos

**Fuente:** dataset público de **Lending Club** (`Loan_status_2007-2020Q3`), que reúne préstamos personales originados entre 2007 y el tercer trimestre de 2020 con su estado final de pago. El diccionario de variables (`LCDataDictionary.xlsx`) acompaña al dataset y se utiliza para clasificar cada feature por el momento en que está disponible (anti-leakage).

**Variable objetivo (`y`):** recodificación de `loan_status` como binaria, descartando préstamos sin desenlace conocido:

| Grupo | Estados | y |
|---|---|---|
| `GOOD_STATES` | Fully Paid (y variantes "Does not meet the credit policy: Fully Paid") | 0 |
| `BAD_STATES` | Charged Off, Default (y variantes "Does not meet the credit policy: Charged Off") | 1 |
| `EXCLUDED_STATES` | Current, In Grace Period, Late (16-30 días), Issued | descartado |

Tras filtrar los estados en transición, la **prevalencia real de la clase positiva (impago) es ≈ 20 %** (≈ 80 % buenos pagadores), reflejo realista del fenómeno. El desbalance se compensa durante el entrenamiento con `class_weight='balanced'` en regresión logística y `scale_pos_weight` en XGBoost.

**Variables (resumen):** datos del solicitante (ingresos, antigüedad laboral, propiedad de vivienda, propósito del préstamo, estado/región), datos del crédito (monto, plazo) y métricas del *credit bureau* (delincuencias, líneas abiertas, utilización, FICO, etc.). Cada variable se etiquetó manualmente como **APP** (disponible al momento de la solicitud) o **post-decisión**, descartando estas últimas para evitar *data leakage*.

## Metodología

El flujo del notebook sigue las fases estándar de un proyecto de *credit scoring*:

**1. Definición del target (anti-leakage).** Se documenta en código por qué cada estado se asigna a `good`, `bad` o se excluye, y se clasifican todas las variables del diccionario por momento de disponibilidad.

**2. Análisis exploratorio y limpieza.** Se descartan variables con más del 50 % de faltantes o sin poder discriminante; se estudia la estructura de los faltantes con `missingno` (¿faltan en las mismas observaciones?, ¿correlacionan con la target?) y se agrupan categorías poco representadas en `home_ownership`, `purpose`, `verification_status`, región y `emp_length`.

**3. Ingeniería de variables.** Las fechas se transforman a **meses de antigüedad** respecto al *snapshot* de evaluación, lo que captura el comportamiento reciente del cliente y resulta uno de los predictores más fuertes. Se crean además features binarias de presencia/ausencia para los faltantes con patrón.

**4. Split estratificado único.** *Train / Validation / Test* en proporción **60 / 20 / 20** estratificado por la target. Los mismos índices se reusan en los tres pipelines de modelado para garantizar comparabilidad.

**5. Selección de variables por IV.** Se seleccionan los predictores con **Information Value > 0.2** (umbral Siddiqi) y se aplica *binning* óptimo con **`optbinning`** (WOE) para tramificar y hacer monotónica cada variable.

**6. Modelado.** Tres modelos sobre los mismos splits:

- **Scorecard (interpretable):** *Logistic Regression* sobre variables WOE-transformadas, escalado a 100–1000 puntos con `optbinning.Scorecard`.
- **XGBoost** y **LightGBM** (challengers): con *early stopping* sobre validación y búsqueda de hiperparámetros con `RandomizedSearchCV` / `GridSearchCV`.

**7. Elección del punto de corte.** Se comparan cuatro criterios sobre validación: *KS máximo*, *F1 máximo*, *Índice de Youden* y **umbral cost-sensitive** parametrizado por `LGD`, `COF`, `OPEX`, `MARGIN`. El criterio **definitivo es el cost-sensitive** (`prob_corte = 0.51` para el scorecard); KS / F1 / Youden quedan como referencias diagnósticas. La justificación: en *credit risk* lo que se minimiza es la **pérdida esperada del portafolio**, no una métrica estadística simétrica — aprobar un mal cliente cuesta mucho más que rechazar a uno bueno.

**8. Calibración de probabilidades.** Como el modelo entrenado con `scale_pos_weight` produce probabilidades infladas (cree que la prevalencia es ≈ 50 % en lugar del ≈ 20 % real), se ajusta un **calibrador isotónico sobre XGBoost** con `CalibratedClassifierCV(cv='prefit')` usando validación. Esto no afecta el ranking (AUC/KS/Gini), pero es **imprescindible para el pricing**, que depende de la probabilidad absoluta. Se evalúa con **Brier score** y **ECE** (Expected Calibration Error).

**9. Pricing por tiers.** Sobre las probabilidades calibradas se construyen bandas de riesgo (grades A → N) y se calcula la tasa por banda con la fórmula:

```
tasa = COF + OPEX + LGD · PD + Capital_Cost + MARGIN
```

con parámetros documentados (`LGD = 0.4`, `COF = 1.5 %`, `OPEX = 2 %`, `MARGIN = 3 %`). La tasa propuesta se compara contra la tasa real de Lending Club por grade.

**10. Empaquetado en pipeline.** *Custom transformers* (`BaseEstimator + TransformerMixin`), `ColumnTransformer`, fit con *early stopping*, envoltura en `CalibratedClassifierCV` y serialización con `joblib`. Se incluye un test de inferencia *cold-start* extremo a extremo (`inference.py`, `score_batch.py`).

## Resultados y Conclusiones

### Métricas en TEST (mismas filas para los tres modelos)

| Métrica | Scorecard (LogReg+WOE) | XGBoost | LightGBM |
|---|---:|---:|---:|
| AUC-ROC | 0.6911 | **0.7109** | 0.6751 |
| AUC-PR | 0.3618 | **0.3861** | 0.3439 |
| KS | 0.2750 | **0.3046** | 0.2509 |
| Gini | 0.3822 | **0.4218** | 0.3501 |
| Recall | 0.6110 | 0.6718 | **0.7336** |
| Precision | 0.3142 | **0.3161** | 0.2739 |
| F1 | 0.4150 | **0.4299** | 0.3989 |
| Brier | 0.2222 | 0.2165 | **0.1598** |
| ECE | 0.2647 | 0.2605 | **0.0290** |
| Threshold (cost-sensitive) | 0.51 | 0.49 | 0.23 |

> **Lectura de la tabla:** XGBoost lidera en discriminación (AUC, KS, Gini, F1) con un *gap* contenido frente al scorecard. LightGBM, aún por debajo en ranking, ofrece la mejor calibración nativa (Brier 0.16, ECE 0.03). El scorecard pierde unos décimos de AUC pero conserva la ventaja decisiva en producción regulatoria: **interpretabilidad total**.

### Conclusiones

**Discriminación.** Todos los modelos discriminan de forma estable y monotónica: los segmentos de score alto presentan probabilidad real de impago mínima y la tasa de mora crece de forma ordenada al bajar de banda — escala de riesgo coherente y auditable.

**Modelo recomendado.** Se priorizan dos modelos complementarios:

- **Scorecard de Logistic Regression + WOE** como modelo principal de decisión por su **interpretabilidad** (cumple los requisitos de explicabilidad regulatoria del *credit scoring* tradicional) y su estabilidad ante volúmenes pequeños.
- **XGBoost calibrado isotónicamente** como *challenger* y como motor de **probabilidades para el pricing**, donde la calibración importa más que la pura interpretabilidad.

**Cut-off.** El criterio elegido es **cost-sensitive** (minimiza la pérdida esperada del portafolio, asumiendo `C_FN ≫ C_FP`), no Youden ni F1. Estos últimos quedan documentados como referencia.

**Pricing.** El esquema de *tiering* permite **tasas competitivas** a los grades superiores (defendiendo cuota de mercado) y **tasas más altas** a los grades inferiores que pasan el corte, rentabilizando la cartera. Las tasas propuestas se contrastan con las observadas en Lending Club por grade.

**Lectura PYME / fintech en expansión.** En empresas de crecimiento agresivo el perfil de la cartera cambia rápido (*riesgo de caducidad rápida*); el modelo debe **monitorearse y recalibrarse cada 3–6 meses**. Con bases históricas pequeñas la simplicidad del scorecard rinde mejor que algoritmos hiper-complejos. El cut-off se vuelve además una **palanca comercial**: se puede flexibilizar conscientemente para captar cuota y endurecer al madurar, conociendo siempre el riesgo extra asumido.

## Stack Tecnológico

- **Manipulación de datos:** `pandas`, `numpy`
- **Modelado tradicional:** `scikit-learn` (`LogisticRegression`, `Pipeline`, `ColumnTransformer`, `CalibratedClassifierCV`, `GridSearchCV`, `RandomizedSearchCV`, `StratifiedKFold`)
- **Scorecard / WOE-IV:** `optbinning` (`Scorecard`, `BinningProcess`, `OptimalBinning`, `ScorecardMonitoring`)
- **Modelos ensemble:** `xgboost` (`XGBClassifier`), `lightgbm` (`LGBMClassifier` con `early_stopping`)
- **Estadística y métricas:** `scipy.stats` (`ks_2samp`); métricas de `sklearn` (ROC-AUC, KS, F1, F-beta, Brier, *precision-recall*) + ECE custom
- **Visualización:** `matplotlib`, `seaborn`, `plotly`, `missingno`
- **Serialización / despliegue:** `joblib`
- **Entorno:** Python 3.10+, Jupyter Notebook

> **Autor:** Samuel García · [GitHub](https://github.com/samgarcia42)  
> Proyecto realizado como parte de un *Challenge* de modelado de riesgo crediticio.
