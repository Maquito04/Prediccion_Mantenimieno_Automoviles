import pandas as pd
import numpy as np
import warnings
import explore
import joblib
from preprocesador import DataPreProcessor 
from trainer import MantenimientoPredictor
from stadistic_analisys import *
warnings.filterwarnings('ignore')

print("Leyendo csv ...")
df = pd.read_csv("car_data.csv")

preprocesador = DataPreProcessor()
# print("Preprocesador hecho")
df_limpio, columnas_categoricas, columnas_numericas = preprocesador.limpiarData(df) 
print("Dataframe limpio")
df_procesado = preprocesador.manejarFaltantes(df_limpio, columnas_numericas, columnas_categoricas)
print("Dataframe procesado")
df_final = preprocesador.encodificar_categoricas(df_procesado, columnas_categoricas)
print("Dataframe codificado")

#Empezar entrenamiento
print("="*80)
print("ENTRENAMIENTO")
print("="*80)
target_column="mpg_combinado"
predictor = MantenimientoPredictor()
X, y = predictor.preparar_caracteristica_objetivo(df_final,target_column)

if X is not None and y is not None:
    if y.dtype == 'object' or y.nunique() < 20:
        modelo, resultados, tipo, X_test, y_test = predictor.entrenar_modelo_clasificacion(X, y)
    else:
        modelo, resultados, tipo, X_test, y_test = predictor.entrenar_modelo_regresion(X, y)

print("Creando pdf ...")
explore.explore_dataset(df,resultados,target_column,tipo, X_test, y_test,columnas_numericas,columnas_categoricas,"explorar.pdf")
print("PDF creado con datos del csv")



