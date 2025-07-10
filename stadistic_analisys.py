import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import (
    pearsonr, spearmanr, shapiro, ttest_ind, chi2_contingency
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import t
from scipy.stats import f_oneway
from itertools import combinations

def calcular_correlaciones(df, target):
    for col in df.columns:
        if col != target:
            try:
                pearson, _ = pearsonr(df[col], df[target])
                spearman, _ = spearmanr(df[col], df[target])
                return f"{col}: Pearson={pearson:.3f}, Spearman={spearman:.3f}"
            except Exception as e:
                return f"{col}: Error -> {e}"

def test_normalidad(df):
    for col in df.columns:
        if df[col].nunique() > 10:
            try:
                stat, p = shapiro(df[col])
                return f"{col}: W={stat:.3f}, p={p:.4f}"
            except:
                return f"{col}: No se pudo aplicar el test."

def calcular_vif(df):
    df = df.dropna()
    if df.empty:
        return "⚠️ No hay suficientes datos numéricos para calcular VIF."
    X = sm.add_constant(df)
    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i+1) for i in range(len(df.columns))]
        return vif_data
    except Exception as e:
        return f"❌ Error al calcular VIF: {e}"

def prueba_ttest(df, col_cont, col_cat):
    categorias = df[col_cat].dropna().unique()
    if len(categorias) == 2:
        grupo1 = df[df[col_cat] == categorias[0]][col_cont]
        grupo2 = df[df[col_cat] == categorias[1]][col_cont]
        stat, p = ttest_ind(grupo1, grupo2)
        return f"t = {stat:.3f}, p = {p:.4f}"
    else:
        return "T-test requiere una variable categórica binaria."

def prueba_chi2(df, col1, col2):
    tabla = pd.crosstab(df[col1], df[col2])
    stat, p, _, _ = chi2_contingency(tabla)
    return f"Chi2 = {stat:.2f}, p = {p:.4f}"

def diebold_mariano(y_true, y_pred1, y_pred2, h=1, loss_function='mse'):
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    if loss_function == 'mse':
        d = (e1 ** 2) - (e2 ** 2)
    elif loss_function == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        return "❌ Funcion de perdida no valida. Usa 'mse' o 'mae'."
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    n = len(d)
    dm_stat = d_mean / np.sqrt((d_var / n))
    p_value = 2 * t.sf(np.abs(dm_stat), df=n-1)
    return f"DM = {dm_stat:.4f}, p = {p_value:.4f} ({'significativo' if p_value < 0.05 else 'no significativo'})"

def generar_matriz_dm(y_true, modelos_dict, loss_function="mse"):
    nombres = list(modelos_dict.keys())
    matriz = pd.DataFrame("", index=nombres, columns=nombres)
    for m1, m2 in combinations(nombres, 2):
        pred1 = modelos_dict[m1]
        pred2 = modelos_dict[m2]
        resultado = diebold_mariano(y_true, pred1, pred2, loss_function=loss_function)
        matriz.loc[m1, m2] = resultado
    return matriz

def prueba_anova(df, col_cont, col_cat):
    grupos = df[col_cat].dropna().unique()
    if len(grupos) < 3:
        return "⚠️ ANOVA requiere al menos 3 grupos."
    try:
        muestras = [df[df[col_cat] == grupo][col_cont].dropna() for grupo in grupos]
        stat, p = f_oneway(*muestras)
        return f"ANOVA F = {stat:.3f}, p = {p:.4f} ({'significativo' if p < 0.05 else 'no significativo'})"
    except Exception as e:
        return f"❌ Error en ANOVA: {e}"
    
def graficar_variables(df, numeric_cols, categorical_cols, ruta_num="./imagenes/variables_numericas.png", ruta_cat="./imagenes/variables_categoricas.png"):
    os.makedirs("./imagenes", exist_ok=True)

    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)

    if n_numeric > 0:
        fig, axes = plt.subplots(2, min(3, n_numeric), figsize=(15, 10))
        fig.suptitle('📊 DISTRIBUCIONES DE VARIABLES NUMÉRICAS', fontsize=14, fontweight='bold')

        axes = axes.flatten() if n_numeric > 1 else [axes]

        for i, col in enumerate(numeric_cols[:6]):  # Máximo 6 gráficos
            if i < len(axes):
                if i < 3:
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribución: {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frecuencia')
                else:
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Boxplot: {col}')

        plt.tight_layout()
        plt.savefig(ruta_num, dpi=300)
        plt.close()

    if n_categorical > 0:
        fig, axes = plt.subplots(1, min(3, n_categorical), figsize=(15, 5))
        fig.suptitle('📈 DISTRIBUCIONES DE VARIABLES CATEGÓRICAS', fontsize=14, fontweight='bold')

        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        for i, col in enumerate(categorical_cols[:3]):  # Máximo 3 gráficos
            if i < len(axes):
                value_counts = df[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=axes[i], color='lightcoral')
                axes[i].set_title(f'Top valores: {col}')
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(ruta_cat, dpi=300)
        plt.close()

def graficar_distribucion_residuales(y_true, y_pred, ruta_imagen="./imagenes/residuales.png"):
    errores = y_true - y_pred
    sns.histplot(errores, kde=True)
    plt.title("Distribución de errores (residuales)")
    plt.xlabel("Error")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(ruta_imagen, dpi=300)
    plt.close()
