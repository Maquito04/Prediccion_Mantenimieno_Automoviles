import streamlit as st
import pandas as pd
import joblib
import base64

# Título de la app
st.title("🔮 Aplicación de Predicción con Modelo ML")

# Cargar modelo
@st.cache_resource
def cargar_modelo(ruta):
    return joblib.load(ruta)

modelo = cargar_modelo("./modelos/Regresion/modelo_Regresion_GradientBoosting.pkl")

st.subheader("📄 Ver informe PDF")
if st.button("Abrir PDF"):
    with open("explorar.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Subida de archivo o entrada manual
st.subheader("📥📁 Cargar archivo CSV")

archivo = st.file_uploader("Carga tu archivo CSV", type=["csv"])
if archivo is not None:
    df = pd.read_csv(archivo)
    st.write("Vista previa de los datos:")
    st.dataframe(df)

if st.button("Hacer Predicción"):
    predicciones = modelo.predict(df)
    st.write("🔍 Resultados de la predicción:")
    st.write("Predicción: ",predicciones)

st.subheader("✅ Resultados de modelos entrenados")  
results = joblib.load('./modelos/resultados_predictor_regresion.pkl')
model_names = list(results.keys())
selected_model = st.selectbox("Ver", model_names)
if selected_model:
    result = results[selected_model]
    st.write(f"🔹 Promedio error absoluto (MAE): `{result['promedioError']:.2f}`")
    st.write(f"🔹 Precisión (R²): `{result['coeficienteError']:.2f}`")
# Pie de página
st.markdown("---")
st.markdown("📘 App creada con Streamlit")
