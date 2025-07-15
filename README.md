---
# MODELO DE PREDICCION DE MANTENIMIENTO DE AUTOMÓVILES
---

## 🧠 Características del Proyecto

- 🔍 **Análisis exploratorio automatizado** (estadísticas, correlaciones, gráficos)
- 📊 **Pruebas estadísticas** (MAE, RMSE, R2 ,ANOVA ,U de Theil)
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

Asegúrate de tener Python 3.8+ y los paquetes necesarios:

1. Crea un ambiente de python
```bash
python -m venv venv
````

2. Activar el ambiente
```bash
venv\Scripts\activate
````

3. Instalar las dependencias
```bash
pip install -r requirements.txt
````

4. Ejecutar el script principal
```bash
python main.py

````

5. Ejecuta la aplicación web:

```bash
streamlit run app.py
```

3. Se generará automáticamente el archivo `explorar.pdf` con todos los resultados.

---

## 📊 Reporte PDF

El informe incluye:

* Información general del dataset
* Correlaciones
* Gráficos de distribución
* Pruebas estadísticas:
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
| MAE, RMSE, R2   | Metricas de los modelos creados
| ANOVA           | Comparación de medias entre varios grupos         |
| U de Theil      | Elección de modelos de regresión                  |


```
