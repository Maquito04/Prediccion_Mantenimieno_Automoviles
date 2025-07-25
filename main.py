import pandas as pd
import numpy as np
import warnings
import joblib
from preprocesador import DataPreProcessor 
from trainer import MantenimientoPredictor
from stadistic_analisys import *
warnings.filterwarnings('ignore')

print("Leyendo csv ...")
df = pd.read_csv("./data/car_data.csv")

preprocesador = DataPreProcessor()
# print("Preprocesador hecho")
df_limpio, columnas_categoricas, columnas_numericas = preprocesador.limpiarData(df) 
print("Dataframe limpio")
df_procesado = preprocesador.manejarFaltantes(df_limpio, columnas_numericas, columnas_categoricas)
print("Dataframe procesado")
df_final = preprocesador.encodificar_categoricas(df_procesado, columnas_categoricas)
joblib.dump(preprocesador, "./modelos/preprocesador.pkl")
print("Dataframe codificado")

#Empezar entrenamiento
print("="*80)
print("ENTRENAMIENTO")
print("="*80)
target_column="mpg_combinado"
predictor = MantenimientoPredictor()
X, y = predictor.preparar_caracteristica_objetivo(df_final,target_column)
joblib.dump(X.columns.tolist(), "./modelos/Regresion/columnas_entrenamiento.pkl")
print("Columnas del entrenamiento guardadas.")

if X is not None and y is not None:
    if y.dtype == 'object' or y.nunique() < 20:
        resultados, tipo, X_test, y_test = predictor.entrenar_modelo_clasificacion(X, y)
    else:
        resultados, tipo, X_test, y_test = predictor.entrenar_modelo_regresion(X, y)

diccionario_data = {
    "target_column": target_column,
    "tipo": tipo,
    "X_test": X_test,
    "y_test": y_test,
    "columnas_numericas": columnas_numericas,
    "columnas_categoricas": columnas_categoricas
}

joblib.dump(diccionario_data, "./modelos/Regresion/diccionario_data.pkl")
print("Diccionario de datos guardado.")



