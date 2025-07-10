import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataPreProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}

    def limpiarData(self, df):
        df_limpio = df.copy()
        duplicates = df_limpio.duplicated().sum()
        if duplicates > 0:
            df_limpio = df_limpio.drop_duplicates()
        columnas_numericas = df_limpio.select_dtypes(include=["int64", "float64"]).columns.tolist()
        columnas_categoricas = df_limpio.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        return df_limpio, columnas_categoricas, columnas_numericas
    
    def manejarFaltantes(self, df, columnas_numericas, columnas_categoricas):
        df_imputado = df.copy()
        #Imputar con Mediana
        for col in columnas_numericas:
            if df_imputado[col].isnull().sum() > 0:
                imputer = SimpleImputer(strategy = 'median')
                df_imputado[col] = imputer.fit_transform(df_imputado[[col]]).flatten()
                self.imputers[col] = imputer
        #Imputar con Moda
        for col in columnas_categoricas:
            if df_imputado[col].isnull().sum() > 0:
                valor_moda = df_imputado[col].astype(str).mode()[0] if not df_imputado[col].astype(str).mode().empty else 'Unknown'
                df_imputado[col].fillna(valor_moda, inplace=True)
        return df_imputado
    
    def encodificar_categoricas(self, df, columnas_categoricas):
        df_encodificado = df.copy()
        for col in columnas_categoricas:
            df_encodificado[col] = df_encodificado[col].astype(str)
            unique_values = df_encodificado[col].nunique()
            if unique_values <= 10:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded_cols = encoder.fit_transform(df_encodificado[[col]])
                feature_names = [f"{col}_{str(category)}" for category in encoder.categories_[0][1:]]

                encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df_encodificado.index)
                df_encodificado = pd.concat([df_encodificado.drop(col, axis=1), encoded_df], axis=1)
                self.encoders[col] = encoder
            else:
                encoder = LabelEncoder()
                df_encodificado[f"{col}_encoded"] = encoder.fit_transform(df_encodificado[col])
                df_encodificado.drop(col, axis=1, inplace=True)
                self.encoders[col] = encoder
        return df_encodificado
        



