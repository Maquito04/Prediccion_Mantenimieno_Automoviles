import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from stadistic_analisys import *
from fpdf import FPDF

def explore_dataset(df, resultados,target_column,tipo, X_test, y_test, col_num, col_cat,output_file="explorar.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=10)

    def write_line(text):
        pdf.multi_cell(0, 5, txt=text)

    write_line("=" * 80)
    write_line(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
    write_line(f"Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    dtypes_counts = df.dtypes.value_counts()
    write_line("Tipo de datos:")
    for dtype, count in dtypes_counts.items():
        write_line(f"  {dtype}: {count}")

    write_line(f"valores duplicados: {df.duplicated().sum()}")

    def generar_heatmap_correlacion(df, ruta_imagen="./imagenes/correlacion.png"):
        plt.figure(figsize=(15, 15))
        matriz_corr = df.corr(numeric_only=True)
        sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300)
        plt.close()

    write_line("=" * 80)
    pdf.set_font("Courier", style="B",size=10)
    write_line("Matriz de correlación")
    pdf.set_font("Courier", style="",size=10)
    generar_heatmap_correlacion(df)
    pdf.image("./imagenes/correlacion.png", w=100)
    
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
            mensaje = f"{i:2d}. {col}: {val} ({pct:.2f}%)" if val > 0 else f"{i:2d}. {col}: sin valores faltantes"
            write_line(mensaje)

    listar_columnas_por_tipo(df, ["int64", "float64"], "Columnas Numéricas:")
    listar_columnas_por_tipo(df, ["object", "category", "bool"], "Columnas Categóricas:")

    graficar_variables(df, col_num, col_cat)
    pdf.set_font("Courier", style="B", size=10)
    write_line("Distribuciones de Variables Numéricas")
    pdf.image("./imagenes/variables_numericas.png", w=180)
    write_line("Distribuciones de Variables Categóricas")
    pdf.image("./imagenes/variables_categoricas.png", w=180)


    write_line("=" * 80)
    pdf.set_font("Courier", style="B",size=10)
    write_line("Datos Estadísticos")
    pdf.set_font("Courier", style="",size=10)
    
    df_num = df.select_dtypes(include=["int64", "float64"]).dropna()
    write_line(calcular_correlaciones(df_num, target_column))
    write_line(test_normalidad(df_num))
    write_line(calcular_vif(df_num.drop(columns=[target_column])).to_string())
    write_line(prueba_ttest(df, target_column, "cilindrada"))
    write_line(prueba_chi2(df, "transmision", "cilindros"))

    write_line("=" * 80)
    pdf.set_font("Courier", style="B",size=10)
    write_line("Resultados de Modelos entrenados")
    pdf.set_font("Courier", style="",size=10)

    def imprimir_resultados_modelos(resultados,tipo):
        for nombre, info in resultados.items():
            write_line(f"Modelo: {nombre}")
            if tipo == "clasificacion":
                write_line(f"  Precisión: {info['precision']:.4f}")
            elif tipo == "regresion":
                write_line(f"  MAE: {info['promedioError']:.4f}")
                write_line(f"  RMSE: {info['raizErrorCuadratico']:.4f}")
                write_line(f"  R²: {info['coeficienteError']:.4f}")

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
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        tabla = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(0.8, 1.0)
        plt.savefig(ruta_imagen, dpi=300)
        plt.close()

    exportar_matriz_como_imagen(matriz_dm)
    pdf.image("./imagenes/matriz_dm.png", w=180)

    anova_result = limpiar_texto(prueba_anova(df, "mpg_combinado", "conduccion"))
    write_line("Prueba ANOVA:\n" + anova_result)
    
    write_line("=" * 80)
    pdf.set_font("Courier", style="B", size=10)
    write_line("Distribución de errores (Gradient Boosting)")
    pdf.set_font("Courier", style="", size=10)
    graficar_distribucion_residuales(y_test, y_pred_modeloGB, "./imagenes/residuales_GB.png")
    pdf.image("./imagenes/residuales_GB.png", w=180)

    write_line("=" * 80)
    pdf.set_font("Courier", style="B", size=10)
    write_line("Distribución de errores (Random Forest)")
    pdf.set_font("Courier", style="", size=10)
    graficar_distribucion_residuales(y_test, y_pred_modeloRF, "./imagenes/residuales_RF.png")
    pdf.image("./imagenes/residuales_RF.png", w=180)

    write_line("=" * 80)
    pdf.set_font("Courier", style="B", size=10)
    write_line("Distribución de errores (Decision Tree)")
    pdf.set_font("Courier", style="", size=10)
    graficar_distribucion_residuales(y_test, y_pred_modeloDT, "./imagenes/residuales_RF.png")
    pdf.image("./imagenes/residuales_RF.png", w=180)

    pdf.output(output_file)


