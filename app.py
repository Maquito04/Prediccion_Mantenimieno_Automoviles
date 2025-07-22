import streamlit as st
import pandas as pd
import joblib
import base64
from preprocesador import DataPreProcessor
from trainer import MantenimientoPredictor
from diccionario_idiomas import *
import explore as ex

st.set_page_config(page_title="Predicción de Mantenimiento", layout="wide")

# Idiomas
idioma = st.selectbox("🌐 Selecciona el idioma / Select language / Sélectionnez la langue:", ("Español", "English", "Français"))

tr = diccionario(idioma)

st.title(tr["titulo"])
st.markdown("---")

#Ver MANUAL
pdf_por_idioma = {
    "Español": "./pdf/manual_español.pdf",
    "English": "./pdf/manual_ingles.pdf",
    "Français": "./pdf/manual_frances.pdf"
}
ruta_manual = pdf_por_idioma.get(idioma)

if "mostrar_manual" not in st.session_state:
    st.session_state.mostrar_manual = False

if st.button(tr["abrir_manual"] if not st.session_state.mostrar_manual else tr["cerrar_manual"]):
    st.session_state.mostrar_manual = not st.session_state.mostrar_manual
if st.session_state.mostrar_manual:
    try:
        with open(ruta_manual, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        manual_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(manual_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(tr["error"] + f" No se encontró el manual para el idioma: {idioma}")

# Cargar modelo y preprocesador
modelo_GB = joblib.load("./modelos/Regresion/modelo_Regresion_GradientBoosting.pkl")
modelo_RF = joblib.load("./modelos/Regresion/modelo_Regresion_RandomForest.pkl")
modelo_DT = joblib.load("./modelos/Regresion/modelo_Regresion_DecisionTree.pkl")

def prediccion_hibrido(X):
    return 0.5*modelo_GB.predict(X) + 0.3*modelo_RF.predict(X) + 0.2*modelo_DT.predict(X)

columnas_entrenamiento = joblib.load("./modelos/Regresion/columnas_entrenamiento.pkl")
preprocesador = joblib.load("./modelos/preprocesador.pkl")

# Ver documentacion
pdf_por_idioma = {
    "Español": "./pdf/explorar_español.pdf",
    "English": "./pdf/explorar_ingles.pdf",
    "Français": "./pdf/explorar_frances.pdf"
}
ruta_pdf = pdf_por_idioma.get(idioma)

if "mostrar_pdf" not in st.session_state:
    st.session_state.mostrar_pdf = False

if st.button(tr["abrir_pdf"] if not st.session_state.mostrar_pdf else tr["cerrar_pdf"]):
    st.session_state.mostrar_pdf = not st.session_state.mostrar_pdf
if st.session_state.mostrar_pdf:
    try:
        with open(ruta_pdf, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        ex.explore_dataset(idioma, ruta_pdf)
        with open(ruta_pdf, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Selector de modo de entrada
modo_entrada = st.radio(tr["modo_entrada"], (tr["archivo_csv"], tr["entrada_manual"]))

# CSV
if modo_entrada == tr["archivo_csv"]:
    archivo = st.file_uploader(tr["subir_csv"], type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)
        st.subheader(tr["vista_previa"])
        st.dataframe(df)

        df_limpio, cat_cols, num_cols = preprocesador.limpiarData(df)
        df_proc = preprocesador.manejarFaltantes(df_limpio, num_cols, cat_cols)
        df_final = preprocesador.encodificar_categoricas(df_proc, cat_cols)

        X = df_final.drop(columns=["mpg_combinado"]) if "mpg_combinado" in df_final.columns else df_final.copy()
        for col in columnas_entrenamiento:
            if col not in X.columns:
                X[col] = 0
        X = X[columnas_entrenamiento]

        if st.button(tr["prediccion_modelo"]):
            try:
                predicciones = prediccion_hibrido(X)
                st.success(tr["resultados"])
                for i, pred in enumerate(predicciones):
                    st.write(tr["prediccion_exitosa"].format(pred=pred))
            except Exception as e:
                st.error(f"{tr['error']} {e}")

# Entrada manual
elif modo_entrada == tr["entrada_manual"]:
    st.subheader(tr["ingreso_manual"])

    def selectbox(label, opciones):
        seleccion = st.selectbox(label, opciones + ["Otro"])
        if seleccion == "Otro":
            return st.text_input(f"{label} - Otro")
        return seleccion

    col1, col2, col3, col4 = st.columns(4)
    with col1: marca = selectbox(tr["marca"], ["Toyota", "Honda", "Ford", "Chevrolet", "Hyundai"])
    with col2: modelo_vehiculo = st.text_input(tr["modelo"])
    with col3: clase = selectbox(tr["clase"], ["SUV", "Sedan", "Hatchback", "Pickup", "Deportivo"])
    with col4: transmision = selectbox(tr["transmision"], ["Automática", "Manual"])

    col5, col6, col7 = st.columns(3)
    with col5: ano = st.number_input(tr["ano"], min_value=1990, max_value=2025, value=2020)
    with col6: cilindros = st.slider(tr["cilindros"], min_value=2, max_value=12, value=4)
    with col7: cilindrada = st.slider(tr["cilindrada"], min_value=1, max_value=10, value=2)

    col8, col9, col10, col11 = st.columns(4)
    with col8: conduccion = selectbox(tr["conduccion"], ["awd", "fwd", "rwd", "4wd"])
    with col9: tipo_combustible = selectbox(tr["tipo_combustible"], ["gas", "diesel", "electricity"])
    with col10: mpg_ciudad = st.number_input(tr["mpg_ciudad"], min_value=0.0)
    with col11: mpg_carretera = st.number_input(tr["mpg_carretera"], min_value=0.0)

    df_manual = pd.DataFrame([{
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
    }])

    st.dataframe(df_manual)

    df_limpio, cat_cols, num_cols = preprocesador.limpiarData(df_manual)
    df_proc = preprocesador.manejarFaltantes(df_limpio, num_cols, cat_cols)
    df_final = preprocesador.encodificar_categoricas(df_proc, cat_cols)

    X = df_final.drop(columns=["mpg_combinado"]) if "mpg_combinado" in df_final.columns else df_final.copy()
    for col in columnas_entrenamiento:
        if col not in X.columns:
            X[col] = 0
    X = X[columnas_entrenamiento]

    if st.button(tr["prediccion_modelo"]):
        try:
            predicciones = prediccion_hibrido(X)
            st.success(tr["resultados"])
            st.write(tr["prediccion_exitosa"].format(pred=predicciones[0]))
        except Exception as e:
            st.error(f"{tr['error']} {e}")

# Pie de página
st.markdown("---")
st.markdown(tr["pie_pagina"])
