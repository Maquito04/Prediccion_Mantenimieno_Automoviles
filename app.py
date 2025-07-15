import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np
import os
from preprocesador import DataPreProcessor 
from trainer import MantenimientoPredictor

st.set_page_config(page_title="Predicción de Mantenimiento", layout="wide")

st.title("🔮 Predicción de mantenimiento de vehículos")
st.markdown("---")

# Ver PDF
st.write("📄 Ver informe PDF")
if st.button("Abrir PDF"):
    with open("explorar.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

#CARGA DE DATOS
modelo = joblib.load("./modelos/Regresion/modelo_Regresion_GradientBoosting.pkl")
columnas_entrenamiento = joblib.load("./modelos/Regresion/columnas_entrenamiento.pkl")
preprocesador = joblib.load("./modelos/preprocesador.pkl")

#INICIO DE INTERFAZ WEB
modo_entrada = st.radio(
    "Selecciona el modo de ingreso de datos:",
    ("Cargar archivo CSV", "Entrada manual")
)

#Entrada manual
if modo_entrada == "Cargar archivo CSV":
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)
        st.subheader("📄 Vista previa de los datos:")
        st.dataframe(df)

        df_limpio, columnas_categoricas, columnas_numericas = preprocesador.limpiarData(df)
        df_procesado = preprocesador.manejarFaltantes(df_limpio, columnas_numericas, columnas_categoricas)
        df_final = preprocesador.encodificar_categoricas(df_procesado, columnas_categoricas)

        predictor = MantenimientoPredictor()
        target_column = "mpg_combinado" if "mpg_combinado" in df_final.columns else None
        
        X = df_final.drop(columns=[target_column]) if target_column else df_final.copy()
        for col in columnas_entrenamiento:
            if col not in X.columns:
                X[col] = 0  
        X = X[columnas_entrenamiento]

        if st.button("Hacer Predicción con modelo por defecto"):
            try:
                predicciones = modelo.predict(X)
                st.success("🔍 Resultados:")
                for i, pred in enumerate(predicciones):
                    st.write(f"Predicción #{i+1}: {pred:.4f} millas por galón combinados")
            except Exception as e:
                st.error(f"❌ Error en la predicción: {e}")

# Entrada manual
elif modo_entrada == "Entrada manual":
    st.subheader("✍️ Ingreso de datos manual")

    def selectbox(label, opciones):
        seleccion = st.selectbox(label, opciones + ["Otro"])
        if seleccion == "Otro":
            valor_otro = st.text_input(f"Ingrese otra opción para {label}")
            return valor_otro
        else:
            return seleccion
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        marca = selectbox("Marca", ["Toyota", "Honda", "Ford", "Chevrolet", "Hyundai"])
    with col2:
        modelo_vehiculo = st.text_input("Modelo")
    with col3:
        clase = selectbox("Clase", ["SUV", "Sedan", "Hatchback", "Pickup", "Deportivo"])
    with col4:
        transmision = selectbox("Transmisión", ["Automática", "Manual"])

    col5, col6, col7 = st.columns(3)
    with col5:
        ano = st.number_input("Año", min_value=1990, max_value=2025, value=2020)
    with col6:
        cilindros = st.slider("Cilindros", min_value=2, max_value=12, value=4)
    with col7:
        cilindrada = st.slider("Cilindrada", min_value=1, max_value=10, value=2)

    col8, col9, col10, col11 = st.columns(4)
    with col8:
        conduccion = selectbox("Conducción", ["awd", "fwd", "rwd", "4wd"])
    with col9:
        tipo_combustible = selectbox("Tipo de combustible", ["gas", "diesel", "electricity"])
    with col10:
        mpg_ciudad = st.number_input("Millas por galón en ciudad", min_value=0.0)
    with col11:
        mpg_carretera = st.number_input("Millas por galón en carretera", min_value=0.0)

    df_manual = {
            "mpg_ciudad": mpg_ciudad,
            "clase": clase,
            "cilindros": cilindros,
            "cilindrada": cilindrada,
            "conduccion": conduccion,
            "tipo_combustible": tipo_combustible,
            "mpg_carretera": mpg_carretera,
            "marca": marca,
            "modelo": modelo_vehiculo,
            "transmision": transmision,
            "ano": ano,
        }
    df_manual = pd.DataFrame([df_manual])  # 👈 Esta línea es clave
    st.dataframe(df_manual)
    df_limpio, columnas_categoricas, columnas_numericas = preprocesador.limpiarData(df_manual)
    df_procesado = preprocesador.manejarFaltantes(df_limpio, columnas_numericas, columnas_categoricas)
    df_final = preprocesador.encodificar_categoricas(df_procesado, columnas_categoricas)

    predictor = MantenimientoPredictor()
    target_column = "mpg_combinado" if "mpg_combinado" in df_final.columns else None
        
    X = df_final.drop(columns=[target_column]) if target_column else df_final.copy()
    for col in columnas_entrenamiento:
        if col not in X.columns:
            X[col] = 0  
    X = X[columnas_entrenamiento]

    if st.button("Hacer Predicción con modelo por defecto"):
        try:
            predicciones = modelo.predict(X)
            st.success("🔍 Resultados:")
            st.write(f"✅ {predicciones[0]:.4f} millas por galón combinados")
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")

# # Pie de página
st.markdown("---")
st.markdown("📘Predicción de Mantenimiento de Automóviles")
