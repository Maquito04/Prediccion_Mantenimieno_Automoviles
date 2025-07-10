---
# MODELO DE PREDICCION DE MANTENIMIENTO DE AUTOMÓVILES
---

## 🧠 Características del Proyecto

- 🔍 **Análisis exploratorio automatizado** (estadísticas, correlaciones, gráficos)
- 📊 **Pruebas estadísticas** (Shapiro, ANOVA, t-test, Chi², VIF, Diebold-Mariano)
- 🤖 **Modelos entrenados**:
  - Regresión: Random Forest, Gradient Boosting, Decision Tree
  - Clasificación (opcional): Random Forest, Gradient Boosting, Logistic Regression
- 📄 **Reporte automático en PDF** con gráficos y resultados
- 📈 **Visualización de errores residuales**

---

## 📁 Estructura del Proyecto

```

├── main.py                     # Script principal
├── explore.py                 # Exploración y generación de PDF
├── preprocesador.py          # Limpieza y codificación de datos
├── trainer.py                # Entrenamiento y evaluación de modelos
├── analysis.py               # Pruebas estadísticas
├── modelos/                  # Modelos guardados (.pkl)
├── resultados/               # Resultados de entrenamiento
├── imagenes/                 # Gráficos generados
└── data/                     # Datasets de entrada

````

---

## ▶️ Ejecución

1. Asegúrate de tener Python 3.8+ y los paquetes necesarios:

```bash
pip install -r requirements.txt
````

2. Ejecuta el script principal:

```bash
python main.py
```

3. Se generará automáticamente el archivo `explorar.pdf` con todos los resultados.

---

## 📊 Reporte PDF

El informe incluye:

* Información general del dataset
* Correlaciones
* Gráficos de distribución
* Pruebas estadísticas:

  * Normalidad (Shapiro)
  * Multicolinealidad (VIF)
  * Chi² y t-test
  * ANOVA
  * Diebold-Mariano
* Métricas de modelos: MAE, RMSE, R²
* Comparaciones entre modelos

---

## 📌 Dependencias

* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `scikit-learn`
* `statsmodels`
* `fpdf`
* `joblib`

---

## 🧪 Pruebas estadísticas incluidas

| Prueba          | Objetivo                                          |
| --------------- | ------------------------------------------------- |
| Shapiro         | Normalidad de variables numéricas                 |
| VIF             | Multicolinealidad entre predictores               |
| T-test          | Comparación de medias entre dos grupos            |
| Chi²            | Asociación entre variables categóricas            |
| ANOVA           | Comparación de medias entre varios grupos         |
| Diebold-Mariano | Comparación de errores entre modelos de regresión |


```
