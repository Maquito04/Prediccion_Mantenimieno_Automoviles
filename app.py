import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np
import os
from preprocesador import DataPreProcessor 
from trainer import MantenimientoPredictor

st.set_page_config(page_title="Predicción de Mantenimiento", layout="wide")

# Título
st.title("🔮 Aplicación de Predicción con Modelo ML")

# Función para cargar modelos
@st.cache_resource
def cargar_modelo(ruta):
    return joblib.load(ruta)

# Cargar modelo por defecto
modelo = cargar_modelo("./modelos/Regresion/modelo_Regresion_GradientBoosting.pkl")

# Ver PDF
st.subheader("📄 Ver informe PDF")
if st.button("Abrir PDF"):
    with open("explorar.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Subida CSV
# st.subheader("📥📁 Cargar archivo CSV para predicción rápida")
# archivo = st.file_uploader("Carga tu archivo CSV", type=["csv"])
# if archivo is not None:
#     df = pd.read_csv(archivo)
#     st.write("Vista previa de los datos:")
#     st.dataframe(df)

#     preprocesador = DataPreProcessor()
#     df_limpio, columnas_categoricas, columnas_numericas = preprocesador.limpiarData(df) 
#     df_procesado = preprocesador.manejarFaltantes(df_limpio, columnas_numericas, columnas_categoricas)
#     df_final = preprocesador.encodificar_categoricas(df_procesado, columnas_categoricas)

#     if st.button("Hacer Predicción con modelo por defecto"):
#         try:
#             predicciones = modelo.predict(df_final)
#             st.success("🔍 Resultados de la predicción:")
#             st.write(predicciones)
#         except Exception as e:
#             st.error(f"❌ Error en la predicción: {e}")

# Selección de modelo
st.subheader("✅ Resultados de modelos entrenados")
results = joblib.load('./resultados/Regresion/resultado_Regresion_GradientBoosting.pkl')
model_names = list(results.keys())
selected_model = st.selectbox("Selecciona un modelo para visualizar métricas y probar", model_names)

if selected_model:
    result = results[selected_model]
    st.write(f"🔹 MAE (Error absoluto medio): `{result['promedioError']:.2f}`")
    st.write(f"🔹 R² (Coef. de determinación): `{result['coeficienteError']:.2f}`")

    modelo_actual = result['modelo']
    if hasattr(modelo_actual, "feature_names_in_"):
        columnas_esperadas = modelo_actual.feature_names_in_
    elif 'features' in result:
        columnas_esperadas = result['features']
    else:
        columnas_esperadas = []

    # ✅ Checkbox para ingreso manual opcional
    if st.checkbox("📝 Ingresar manualmente una fila de prueba"):
        st.subheader("🧪 Fila manual de entrada")
        entrada_manual = {}
        for col in columnas_esperadas:
            entrada_manual[col] = st.number_input(f"Ingrese valor para '{col}'", value=0.0)

        if st.button("🔍 Predecir con fila manual"):
            df_manual = pd.DataFrame([entrada_manual])
            try:
                pred = modelo_actual.predict(df_manual)
                st.success(f"✅ Predicción estimada: {pred[0]:.2f}")
            except Exception as e:
                st.error(f"❌ Error al predecir: {e}")

# Pie de página
st.markdown("---")
st.markdown("📘 App creada con Streamlit | Proyecto de Tesis - Predicción de Mantenimiento de Automóviles")
