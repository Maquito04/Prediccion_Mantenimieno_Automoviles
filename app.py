import streamlit as st
import pandas as pd
import joblib
import base64
from preprocesador import DataPreProcessor
from trainer import MantenimientoPredictor

# Configuración
st.set_page_config(page_title="Predicción de Mantenimiento", layout="wide")

# Idiomas disponibles
idioma = st.selectbox("🌐 Selecciona el idioma / Select language / Sélectionnez la langue:", ("Español", "English", "Français"))

# Diccionario de traducciones
traducciones = {
    "Español": {
        "titulo": "🔮 Predicción de mantenimiento de vehículos",
        "modo_entrada": "Selecciona el modo de ingreso de datos:",
        "archivo_csv": "Cargar archivo CSV",
        "entrada_manual": "Entrada manual",
        "subir_csv": "Sube tu archivo CSV",
        "vista_previa": "📄 Vista previa de los datos:",
        "prediccion_modelo": "Hacer Predicción con modelo por defecto",
        "resultados": "🔍 Resultados:",
        "error": "❌ Error en la predicción:",
        "ingreso_manual": "✍️ Ingreso de datos manual",
        "prediccion_exitosa": "✅ {pred:.4f} millas por galón combinados",
        "ver_pdf": "📄 Ver informe PDF",
        "abrir_pdf": "Abrir PDF",
    },
    "English": {
        "titulo": "🔮 Vehicle Maintenance Prediction",
        "modo_entrada": "Select the data input mode:",
        "archivo_csv": "Upload CSV file",
        "entrada_manual": "Manual input",
        "subir_csv": "Upload your CSV file",
        "vista_previa": "📄 Data preview:",
        "prediccion_modelo": "Make Prediction with default model",
        "resultados": "🔍 Results:",
        "error": "❌ Prediction error:",
        "ingreso_manual": "✍️ Manual data input",
        "prediccion_exitosa": "✅ {pred:.4f} miles per gallon combined",
        "ver_pdf": "📄 View PDF report",
        "abrir_pdf": "Open PDF",
    },
    "Français": {
        "titulo": "🔮 Prédiction de l'entretien des véhicules",
        "modo_entrada": "Sélectionnez le mode de saisie des données :",
        "archivo_csv": "Télécharger un fichier CSV",
        "entrada_manual": "Saisie manuelle",
        "subir_csv": "Téléchargez votre fichier CSV",
        "vista_previa": "📄 Aperçu des données :",
        "prediccion_modelo": "Faire une prédiction avec le modèle par défaut",
        "resultados": "🔍 Résultats :",
        "error": "❌ Erreur lors de la prédiction :",
        "ingreso_manual": "✍️ Saisie manuelle des données",
        "prediccion_exitosa": "✅ {pred:.4f} milles par gallon combinés",
        "ver_pdf": "📄 Voir le rapport PDF",
        "abrir_pdf": "Ouvrir le PDF",
    }
}
tr = traducciones[idioma]

# Título
st.title(tr["titulo"])
st.markdown("---")

# Cargar modelo y preprocesador
modelo = joblib.load("./modelos/Regresion/modelo_Regresion_GradientBoosting.pkl")
columnas_entrenamiento = joblib.load("./modelos/Regresion/columnas_entrenamiento.pkl")
preprocesador = joblib.load("./modelos/preprocesador.pkl")

# Ver PDF
st.write(tr["ver_pdf"])
if st.button(tr["abrir_pdf"]):
    with open("explorar.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
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
                predicciones = modelo.predict(X)
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
    with col1: marca = selectbox("Marca", ["Toyota", "Honda", "Ford", "Chevrolet", "Hyundai"])
    with col2: modelo_vehiculo = st.text_input("Modelo")
    with col3: clase = selectbox("Clase", ["SUV", "Sedan", "Hatchback", "Pickup", "Deportivo"])
    with col4: transmision = selectbox("Transmisión", ["Automática", "Manual"])

    col5, col6, col7 = st.columns(3)
    with col5: ano = st.number_input("Año", min_value=1990, max_value=2025, value=2020)
    with col6: cilindros = st.slider("Cilindros", min_value=2, max_value=12, value=4)
    with col7: cilindrada = st.slider("Cilindrada", min_value=1, max_value=10, value=2)

    col8, col9, col10, col11 = st.columns(4)
    with col8: conduccion = selectbox("Conducción", ["awd", "fwd", "rwd", "4wd"])
    with col9: tipo_combustible = selectbox("Tipo de combustible", ["gas", "diesel", "electricity"])
    with col10: mpg_ciudad = st.number_input("Millas por galón en ciudad", min_value=0.0)
    with col11: mpg_carretera = st.number_input("Millas por galón en carretera", min_value=0.0)

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
            predicciones = modelo.predict(X)
            st.success(tr["resultados"])
            st.write(tr["prediccion_exitosa"].format(pred=predicciones[0]))
        except Exception as e:
            st.error(f"{tr['error']} {e}")

# Pie de página
st.markdown("---")
st.markdown("📘 Predicción de Mantenimiento de Automóviles")
