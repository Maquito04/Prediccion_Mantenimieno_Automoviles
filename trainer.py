import pandas as pd
import numpy as np
import joblib
from preprocesador import DataPreProcessor 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor


class MantenimientoPredictor:
    def __init__(self):
        self.models = {}
        self.preprocesador = DataPreProcessor
        self.feature_importance = {}

    def preparar_caracteristica_objetivo(self, df, columna_objetivo):
        if columna_objetivo not in df.columns:
            print("No se encuentra la columna objetivo")
            return None,None
        X = df.drop(columns=[columna_objetivo])
        y = df[columna_objetivo]
        return X, y
    
    def entrenar_modelo_clasificacion(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        results = {}
        for name, model in models.items():
            print(f"Entrenando {name} ...")
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            precision = accuracy_score(y, y_pred)

            results[name] = {
                'modelo': model,
                'precision': precision,
                'predicciones': y_pred,
                'datos_prueba': (X_test,y_test)
            }
            joblib.dump(model, f"./modelos/Clasificacion/modelo_Clasificacion_{name}.pkl")
            joblib.dump(results, f"./resultados/Clasificacion/resultado_Clasificacion_{name}.pkl")

        mejor_modelo_nombre = max(results.keys(), key=lambda k: results[k]['precision'])
        mejor_modelo = results[mejor_modelo_nombre]['modelo']
        self.models['clasificacion'] = mejor_modelo
        return mejor_modelo, results, "clasificacion", X_test, y_test

    def entrenar_modelo_regresion(self,X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10)
        }
        results = {}
        for name, model in models.items():
            print(f"Entrenando {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {
                'modelo': model,
                'promedioError': mae,
                'coeficienteError': r2,
                'prediccion': y_pred,
                'raizErrorCuadratico': rmse,
                'datos_prueba': (X_test, y_test)
            }
            joblib.dump(model, f"./modelos/Regresion/modelo_Regresion_{name}.pkl")
            joblib.dump(results, f"./resultados/Regresion/resultado_Regresion_{name}.pkl")

        mejor_modelo_nombre = min(results.keys(), key=lambda k: results[k]['promedioError'])
        mejor_modelo = results[mejor_modelo_nombre]['modelo']
        self.models['regresion'] = mejor_modelo
        return mejor_modelo, results, "regresion", X_test, y_test
    
    def predecir_nueva_data(self, new_data):
        if 'clasificacion' in self.models:
            pred_class = self.models['clasificacion'].predict(new_data)
            prob_class = self.models['clasificacion'].predict_proba(new_data)
            return {
                'Clasificacion': pred_class,
                'Probabilidad': prob_class
            }
        elif 'regresion' in self.models:
            pred_reg = self.models['regresion'].predict(new_data)
            return {
                'regresion': pred_reg
            }


