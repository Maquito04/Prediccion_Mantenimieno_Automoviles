import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from stadistic_analisys import *
from fpdf import FPDF
from diccionario_idiomas import *

def explore_dataset(idioma,output_file="./pdf/explorar.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=10)

    tr = diccionario(idioma)

    resultados_rf = joblib.load("./resultados/Regresion/resultado_Regresion_RandomForest.pkl")
    resultados_gb = joblib.load("./resultados/Regresion/resultado_Regresion_GradientBoosting.pkl")
    resultados_dt = joblib.load("./resultados/Regresion/resultado_Regresion_DecisionTree.pkl")

    # Crear diccionario general con todos los resultados combinados (igual que antes)
    resultados = {
        "RandomForest": resultados_rf["RandomForest"],
        "GradientBoosting": resultados_gb["GradientBoosting"],
        "DecisionTree": resultados_dt["DecisionTree"]
    }

    df = df = pd.read_csv("./data/car_data.csv")

    diccionario_data = joblib.load("./modelos/Regresion/diccionario_data.pkl")

    target_column = diccionario_data["target_column"]
    tipo = diccionario_data["tipo"]
    X_test = diccionario_data["X_test"]
    y_test = diccionario_data["y_test"]
    columnas_numericas = diccionario_data["columnas_numericas"]
    columnas_categoricas = diccionario_data["columnas_categoricas"]

    def write_line(text):
        pdf.multi_cell(0, 5, txt=text)

    write_line("=" * 80)
    write_line(f"{tr['dimensiones']}: {df.shape[0]} filas, {df.shape[1]} columnas")
    write_line(f"{tr['memoria_usada']}: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    dtypes_counts = df.dtypes.value_counts()
    write_line(tr["tipo_datos"])
    for dtype, count in dtypes_counts.items():
        write_line(f"  {dtype}: {count}")

    write_line(f"{tr['duplicados']}: {df.duplicated().sum()}")

    def generar_heatmap_correlacion(df, ruta_imagen="./imagenes/correlacion.png"):
        plt.figure(figsize=(10, 6))
        matriz_corr = df.corr(numeric_only=True)
        sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300)
        plt.close()

    write_line("=" * 80)
    pdf.set_font("Courier", style="B",size=10)
    write_line(tr["matriz_correlacion"])
    pdf.set_font("Courier", style="",size=10)
    generar_heatmap_correlacion(df)
    pdf.image("./imagenes/correlacion.png", w=150)
    
    def listar_columnas_por_tipo(df, tipos, titulo):
        write_line("-" * 80)
        pdf.set_font("Courier", style="B", size=10)
        write_line(titulo)
        pdf.set_font("Courier", style="", size=10)

        df_tipo = df.select_dtypes(include=tipos)
        missing_data = df_tipo.isnull().sum().sort_values()
        missing_percent = (missing_data / len(df)) * 100

        for i, col in enumerate(df_tipo.columns, 1):
            val = missing_data[col]
            pct = missing_percent[col]
            if val > 0:
                mensaje = f"{i:2d}. " + tr["faltantes"].format(col=col, val=val, pct=pct)
            else:
                mensaje = f"{i:2d}. " + tr["sin_valores_faltantes"].format(col=col)
            write_line(mensaje)

    listar_columnas_por_tipo(df, ["int64", "float64"], tr["columnas_numericas"])
    listar_columnas_por_tipo(df, ["object", "category", "bool"], tr["columnas_categoricas"])

    pdf.add_page()
    graficar_variables(df, columnas_numericas, columnas_categoricas)
    pdf.set_font("Courier", style="B", size=10)
    write_line(tr["dist_num"])
    pdf.image("./imagenes/variables_numericas.png", w=180)
    write_line(tr["dist_cat"])
    pdf.image("./imagenes/variables_categoricas.png", w=180)

    write_line("=" * 80)
    pdf.set_font("Courier", style="B",size=10)
    write_line(tr["resultados_modelos"])
    pdf.set_font("Courier", style="",size=10)

    def imprimir_resultados_modelos(resultados,tipo):
        for nombre, info in resultados.items():
            write_line(f"{tr['modelo']}: {nombre}")
            if tipo == "clasificacion":
                write_line(f"{tr['precision']}: {info['precision']:.4f}")
            elif tipo == "regresion":
                write_line(f"  {tr['mae']}: {info['promedioError']:.4f}")
                write_line(f"  {tr['rmse']}: {info['raizErrorCuadratico']:.4f}")
                write_line(f"  {tr['r2']}: {info['coeficienteError']:.4f}")

    imprimir_resultados_modelos(resultados, tipo)

    modeloGB = joblib.load("./modelos/Regresion/modelo_Regresion_GradientBoosting.pkl")
    modeloRF = joblib.load("./modelos/Regresion/modelo_Regresion_RandomForest.pkl")
    modeloDT = joblib.load("./modelos/Regresion/modelo_Regresion_DecisionTree.pkl")

    y_pred_modeloGB = modeloGB.predict(X_test)
    y_pred_modeloRF = modeloRF.predict(X_test)
    y_pred_modeloDT = modeloDT.predict(X_test)

    def limpiar_texto(texto):
        return texto.encode('latin-1', 'ignore').decode('latin-1')

    matriz_dm = generar_matriz_dm(y_test, {
        "RandomForest": y_pred_modeloRF,
        "GradientBoosting": y_pred_modeloGB,
        "DecisionTree": y_pred_modeloDT
    })

    def exportar_matriz_como_imagen(df, ruta_imagen="./imagenes/matriz_dm.png"):
        fig, ax = plt.subplots(figsize=(15,6))
        ax.axis('off')
        tabla = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(0.7, 1.2)
        plt.savefig(ruta_imagen, dpi=300)
        plt.close()

    u_lr = theils_u(y_test, y_pred_modeloGB)
    u_rf = theils_u(y_test, y_pred_modeloRF)
    u_dt = theils_u(y_test, y_pred_modeloDT)
    
    pdf.add_page()
    write_line("=" * 80)
    pdf.set_font("Courier", style="B", size=14)
    write_line(tr["pruebas"])
    pdf.set_font("Courier", style="", size=10)
    write_line("=" * 80)
    pdf.set_font("Courier", style="B", size=10)
    write_line(tr["pruebas_u_theil"])
    pdf.set_font("Courier", style="", size=10)
    write_line(f"U Theil - Gradient Boosting: {u_lr}")
    write_line(f"U Theil - Random Forest: {u_rf}")
    write_line(f"U Theil - Decision Tree: {u_dt}")    

    # exportar_matriz_como_imagen(matriz_dm)
    # pdf.image("./imagenes/matriz_dm.png", w=180)

    pdf.set_font("Courier", style="B", size=10)
    write_line(tr["prueba_anova"])
    pdf.set_font("Courier", style="", size=10)
    anova_result = limpiar_texto(prueba_anova(df, "mpg_combinado", "conduccion"))
    write_line(anova_result)
   
    write_line("=" * 80)
    pdf.set_font("Courier", style="B", size=10)
    write_line(f"{tr['distribucion_errores']} (Gradient Boosting)")
    pdf.set_font("Courier", style="", size=10)
    graficar_distribucion_residuales(y_test, y_pred_modeloGB, "./imagenes/residuales_GB.png")
    pdf.image("./imagenes/residuales_GB.png", w=80)

    write_line("=" * 80)
    pdf.set_font("Courier", style="B", size=10)
    write_line(f"{tr['distribucion_errores']} (Random Forest)")
    pdf.set_font("Courier", style="", size=10)
    graficar_distribucion_residuales(y_test, y_pred_modeloRF, "./imagenes/residuales_RF.png")
    pdf.image("./imagenes/residuales_RF.png", w=80)

    write_line("=" * 80)
    pdf.set_font("Courier", style="B", size=10)
    write_line(f"{tr['distribucion_errores']} (Decision Tree)")
    pdf.set_font("Courier", style="", size=10)
    graficar_distribucion_residuales(y_test, y_pred_modeloDT, "./imagenes/residuales_RF.png")
    pdf.image("./imagenes/residuales_RF.png", w=80)

    pdf.output(output_file)


