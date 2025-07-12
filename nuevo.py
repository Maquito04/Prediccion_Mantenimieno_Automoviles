import pandas as pd
import joblib
from preprocesador import DataPreProcessor 
from trainer import MantenimientoPredictor

# Cargar los mismos datos con los que entrenaste el modelo
df = pd.read_csv("sample_4.csv")  # ⚠️ Usa aquí tu archivo real
def cargar_modelo(ruta):
    return joblib.load(ruta)

# Cargar modelo por defecto
modelo = cargar_modelo("./modelos/Regresion/modelo_Regresion_GradientBoosting.pkl")

# Procesamiento igual que en el entrenamiento
preprocesador = DataPreProcessor()
df_limpio, columnas_categoricas, columnas_numericas = preprocesador.limpiarData(df)
df_procesado = preprocesador.manejarFaltantes(df_limpio, columnas_numericas, columnas_categoricas)
df_final = preprocesador.encodificar_categoricas(df_procesado, columnas_categoricas)

predictor = MantenimientoPredictor()
target_column = "mpg_combinado"
X, y = predictor.preparar_caracteristica_objetivo(df_final, target_column)

# Guardar columnas usadas en el entrenamiento
joblib.dump(X.columns.tolist(), "./modelos/Regresion/columnas_entrenamiento.pkl")
columnas_entrenamiento = joblib.load("./modelos/Regresion/columnas_entrenamiento.pkl")

# Asegurar que todas las columnas estén presentes
for col in columnas_entrenamiento:
    if col not in X.columns:
        X[col] = 0

X = X[columnas_entrenamiento]
print(modelo.predict(X))