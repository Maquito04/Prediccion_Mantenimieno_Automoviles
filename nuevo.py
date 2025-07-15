from trainer import MantenimientoPredictor
import pandas as pd
import joblib
from trainer import MantenimientoPredictor

# 1. Cargar datos nuevos
df = pd.read_csv("sample.csv")

# 2. Cargar modelo, columnas y preprocesador
modelo = joblib.load("./modelos/Regresion/modelo_Regresion_GradientBoosting.pkl")
columnas_entrenamiento = joblib.load("./modelos/Regresion/columnas_entrenamiento.pkl")  # ✅ NO regenerar esto
preprocesador = joblib.load("./modelos/preprocesador.pkl")

# 3. Preprocesar datos
df_limpio, columnas_categoricas, columnas_numericas = preprocesador.limpiarData(df)
df_procesado = preprocesador.manejarFaltantes(df_limpio, columnas_numericas, columnas_categoricas)
df_final = preprocesador.encodificar_categoricas(df_procesado, columnas_categoricas)

# 4. Preparar X para predicción
predictor = MantenimientoPredictor()
target_column = "mpg_combinado" if "mpg_combinado" in df_final.columns else None
X = df_final.drop(columns=[target_column]) if target_column else df_final.copy()

# 5. Asegurar que X tenga las columnas esperadas
for col in columnas_entrenamiento:
    if col not in X.columns:
        X[col] = 0  # Agrega columnas faltantes

X = X[columnas_entrenamiento]  # Reordena y filtra

# 6. Predicción
y_pred = modelo.predict(X)
print("🔮 Predicciones:")
for i, val in enumerate(y_pred, 1):
    print(f"Registro {i}: {val:.2f}")