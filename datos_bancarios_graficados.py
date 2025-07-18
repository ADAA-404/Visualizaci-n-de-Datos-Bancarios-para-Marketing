"""
datos_bancarios_graficados.py

Este script está diseñado para la exploración y visualización de un conjunto de datos
relacionado con campañas de telemarketing bancario. Utiliza librerías como pandas,
matplotlib, seaborn y plotly para generar diversas visualizaciones que ayudan a
entender las características de los clientes y el impacto de las campañas.

Nota de los derechos de la base de datos:
Este conjunto de datos está disponible públicamente para investigación. Los detalles
se describen en [Moro et al., 2014]. Por favor, incluya esta cita si planea usar
esta base de datos:
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict
the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31,
June 2014
"""

# --- 1. Descarga y Preparación de Datos ---
import os
import requests
import zipfile

def download_and_extract_data(url: str, filename: str, extract_path: str = "."):
    """
    Descarga un archivo zip desde una URL y lo extrae.

    Args:
        url (str): La URL del archivo zip a descargar.
        filename (str): El nombre del archivo zip una vez descargado.
        extract_path (str): La ruta donde se extraerán los contenidos del zip.
    """
    print(f"Intentando descargar: {url} a {filename}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lanza una excepción para errores HTTP
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Descargado exitosamente: {filename}")

        print(f"Extrayendo {filename} a {extract_path}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extracción completada.")

    except requests.exceptions.RequestException as e:
        print(f"Error al descargar {url}: {e}")
    except zipfile.BadZipFile:
        print(f"Error: El archivo {filename} no es un archivo zip válido.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# URLs de los datasets
DATA_URL_1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
DATA_URL_2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VDA_Banking_L2/bank-additional.zip"
ZIP_FILENAME = "bank-additional.zip"
EXTRACT_FOLDER = "bank-additional"
CSV_FILENAME = os.path.join(EXTRACT_FOLDER, 'bank-additional-full.csv')

# Intenta descargar y extraer de la primera URL, si falla, intenta con la segunda
if not os.path.exists(CSV_FILENAME):
    download_and_extract_data(DATA_URL_1, ZIP_FILENAME)
if not os.path.exists(CSV_FILENAME): # Si la primera URL falló o el archivo no existe
    download_and_extract_data(DATA_URL_2, ZIP_FILENAME)

if not os.path.exists(CSV_FILENAME):
    print(f"ERROR: No se pudo encontrar el archivo {CSV_FILENAME}. Asegúrate de que las URLs de descarga sean correctas o descárgalo manualmente.")
    exit() # Salir si el archivo de datos no está disponible

# --- 2. Declaración e Importación de Librerías ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# Configuración para Jupyter Notebooks
try:
    init_notebook_mode(connected=True)
    # Habilitar matplotlib inline solo si estamos en un entorno interactivo
    if 'ipykernel' in sys.modules:
        get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    print("No se detectó un entorno interactivo (Jupyter/IPython). Algunas funcionalidades interactivas podrían no estar disponibles.")
    pass # No hacer nada si no estamos en un entorno interactivo

# Ajustes generales para visualización
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['font.size'] = 12

# Filtrar advertencias
warnings.filterwarnings('ignore')

# Configurar la precisión de pandas para la visualización
pd.set_option("precision", 2)
pd.options.display.float_format = '{:.2f}'.format

# --- 3. Carga y Preprocesamiento Inicial de Datos ---
print("\n--- 3. Carga y Preprocesamiento Inicial de Datos ---")
df = pd.read_csv(CSV_FILENAME, sep=';')
print("Primeras 5 filas del DataFrame:")
print(df.head(5))

# Mapear la variable objetivo 'y' a valores numéricos (0 para 'no', 1 para 'yes')
d = {"no": 0, "yes": 1}
df["y"] = df["y"].map(d)

print("\nColumnas del DataFrame:")
print(df.columns)

print("\nForma del DataFrame (filas, columnas):")
print(df.shape)

# --- 4. Visualización de Datos con Matplotlib ---
print("\n--- 4. Visualización de Datos con Matplotlib ---")

print("\nHistograma de la edad:")
df["age"].hist()
plt.title("Distribución de Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

print("\nEdad promedio por estado civil (línea):")
df[["age", "marital"]].groupby("marital").mean().plot()
plt.title("Edad Promedio por Estado Civil")
plt.ylabel("Edad Promedio")
plt.xlabel("Estado Civil")
plt.show()

print("\nEdad promedio por estado civil (barras):")
df[["age", "marital"]].groupby("marital").mean().plot(kind="bar", rot=45)
plt.title("Edad Promedio por Estado Civil")
plt.ylabel("Edad Promedio")
plt.xlabel("Estado Civil")
plt.tight_layout()
plt.show()

# --- 5. Visualización de Datos con Seaborn ---
print("\n--- 5. Visualización de Datos con Seaborn ---")

print("\nMatriz de gráficos de dispersión (pairplot) para edad, duración y campaña:")
sns.pairplot(df[["age", "duration", "campaign"]])
plt.suptitle("Pair Plot de Edad, Duración y Campaña", y=1.02) # Ajustar título
plt.show()

print("\nDistribución de la edad (distplot/histplot):")
# distplot está deprecado, se recomienda histplot o displot
sns.histplot(df.age, kde=True)
plt.title("Distribución de Edad")
plt.xlabel("Edad")
plt.ylabel("Densidad/Frecuencia")
plt.show()

print("\nGráfico conjunto (jointplot) de edad vs duración:")
sns.jointplot(x="age", y="duration", data=df, kind="scatter")
plt.suptitle("Edad vs Duración de la Llamada", y=1.02)
plt.show()

print("\nBoxplot de edad por los 5 trabajos principales:")
top_jobs = df.job.value_counts().sort_values(ascending=False).head(5).index.values
sns.boxplot(y="job", x="age", data=df[df.job.isin(top_jobs)], orient="h")
plt.title("Distribución de Edad por Tipo de Trabajo (Top 5)")
plt.xlabel("Edad")
plt.ylabel("Tipo de Trabajo")
plt.tight_layout()
plt.show()

print("\nMapa de calor (heatmap) de la suma de suscripciones por trabajo y estado civil:")
job_marital_y = df.pivot_table(index="job", columns="marital", values="y", aggfunc=sum)
sns.heatmap(job_marital_y, annot=True, fmt="d", linewidths=0.5, cmap="viridis")
plt.title("Suscripciones (y=1) por Trabajo y Estado Civil")
plt.xlabel("Estado Civil")
plt.ylabel("Tipo de Trabajo")
plt.tight_layout()
plt.show()

# --- 6. Visualización Interactiva con Plotly ---
print("\n--- 6. Visualización Interactiva con Plotly ---")

print("\nEstadísticas de suscripciones por edad (interactivo):")
age_df = (
    df.groupby("age")[["y"]]
    .sum()
    .join(df.groupby("age")[["y"]].count(), rsuffix='_count')
)
age_df.columns = ["Attracted", "Total Number"]

trace0 = go.Scatter(x=age_df.index, y=age_df["Attracted"], name="Atraídos", mode='lines+markers')
trace1 = go.Scatter(x=age_df.index, y=age_df["Total Number"], name="Número Total", mode='lines+markers')

data = [trace0, trace1]
layout = {"title": "Estadísticas de Suscripciones por Edad del Cliente",
          "xaxis_title": "Edad",
          "yaxis_title": "Conteo"}

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False) # show_link=False evita el botón "Export to plot.ly"

print("\nSuscripciones por mes (interactivo):")
month_index = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month_df = (
    df.groupby("month")[["y"]]
    .sum()
    .join(df.groupby("month")[["y"]].count(), rsuffix='_count')
).reindex(month_index)
month_df.columns = ["Attracted", "Total Number"]

trace0 = go.Bar(x=month_df.index, y=month_df["Attracted"], name="Atraídos")
trace1 = go.Bar(x=month_df.index, y=month_df["Total Number"], name="Número Total")

data = [trace0, trace1]
layout = {"title": "Suscripciones por Mes",
          "xaxis_title": "Mes",
          "yaxis_title": "Conteo"}

fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)

print("\nBoxplot interactivo de edad por estado civil:")
data_box = []
for status in df.marital.unique():
    data_box.append(go.Box(y=df[df.marital == status].age, name=status))
layout_box = {"title": "Distribución de Edad por Estado Civil"}
fig_box = go.Figure(data=data_box, layout=layout_box)
iplot(fig_box, show_link=False)

# --- 7. Análisis de Variables (Numéricas y Categóricas) ---
print("\n--- 7. Análisis de Variables (Numéricas y Categóricas) ---")

print("\nHistograma de edad (solo una variable numérica):")
df["age"].hist()
plt.title("Distribución de Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

print("\nBoxplot de 'cons.price.idx':")
sns.boxplot(x=df["cons.price.idx"])
plt.title("Boxplot de Índice de Precios al Consumidor")
plt.xlabel("Índice de Precios al Consumidor")
plt.show()

print("\nConteo de valores para la variable 'marital':")
print(df["marital"].value_counts().head())

print("\nConteo de valores para la variable objetivo 'y':")
print(df["y"].value_counts())

print("\nConteo de suscripciones (y) usando Seaborn:")
sns.countplot(x=df["y"])
plt.title("Conteo de Suscripciones (0: No, 1: Sí)")
plt.xlabel("Suscripción")
plt.ylabel("Conteo")
plt.show()

print("\nConteo de estado civil (marital) usando Seaborn:")
sns.countplot(x=df["marital"])
plt.title("Conteo de Estado Civil")
plt.xlabel("Estado Civil")
plt.ylabel("Conteo")
plt.show()

print("\nConteo de los 5 trabajos principales:")
plot = sns.countplot(x=df[df["job"].isin(df["job"].value_counts().head(5).index)]["job"])
plt.title("Conteo de los 5 Trabajos Principales")
plt.xlabel("Trabajo")
plt.ylabel("Conteo")
plt.setp(plot.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()

# --- 8. Interacciones entre Variables Numéricas ---
print("\n--- 8. Interacciones entre Variables Numéricas ---")

feat_numeric_interaction = ["cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

print("\nHistogramas de variables económicas seleccionadas:")
df[feat_numeric_interaction].hist(figsize=(10, 8), bins=30)
plt.suptitle("Histogramas de Variables Económicas Seleccionadas", y=1.02)
plt.tight_layout()
plt.show()

print("\nPairplot de variables económicas seleccionadas:")
sns.pairplot(df[feat_numeric_interaction])
plt.suptitle("Pair Plot de Variables Económicas Seleccionadas", y=1.02)
plt.show()

print("\nMapa de calor de la correlación entre variables económicas seleccionadas:")
sns.heatmap(df[feat_numeric_interaction].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación de Variables Económicas")
plt.tight_layout()
plt.show()

# --- 9. Interacciones entre Variables Numéricas y Categóricas ---
print("\n--- 9. Interacciones entre Variables Numéricas y Categóricas ---")

print("\nBoxplot de edad por suscripción (y):")
sns.boxplot(x="y", y="age", data=df)
plt.title("Distribución de Edad por Suscripción")
plt.xlabel("Suscripción (0: No, 1: Sí)")
plt.ylabel("Edad")
plt.show()

print("\nBoxplot de edad por estado civil:")
sns.boxplot(x="marital", y="age", data=df)
plt.title("Distribución de Edad por Estado Civil")
plt.xlabel("Estado Civil")
plt.ylabel("Edad")
plt.show()

print("\nViolinplot de edad por suscripción (y):")
sns.violinplot(x="y", y="age", data=df)
plt.title("Distribución de Edad por Suscripción (Violin Plot)")
plt.xlabel("Suscripción (0: No, 1: Sí)")
plt.ylabel("Edad")
plt.show()

print("\nEdad promedio por situación de vivienda:")
print(df.groupby("housing")["age"].mean())

print("\nBoxplot de edad por situación de vivienda:")
sns.boxplot(x="housing", y="age", data=df)
plt.title("Distribución de Edad por Situación de Vivienda")
plt.xlabel("Situación de Vivienda")
plt.ylabel("Edad")
plt.show()

# --- 10. Interacciones entre Variables Categóricas ---
print("\n--- 10. Interacciones entre Variables Categóricas ---")

print("\nTabla de contingencia: Suscripción (y) vs Estado Civil:")
print(pd.crosstab(df["y"], df["marital"]))

print("\nConteo de estado civil por suscripción (y):")
sns.countplot(x="marital", hue="y", data=df)
plt.title("Conteo de Estado Civil por Suscripción")
plt.xlabel("Estado Civil")
plt.ylabel("Conteo")
plt.legend(title="Suscripción", labels=["No", "Sí"])
plt.show()

print("\nConteo de mes por suscripción (y):")
sns.countplot(x="month", hue="y", data=df, order=month_index) # Usar month_index para ordenar
plt.title("Conteo de Mes por Suscripción")
plt.xlabel("Mes")
plt.ylabel("Conteo")
plt.legend(title="Suscripción", labels=["No", "Sí"])
plt.tight_layout()
plt.show()

# --- 11. Análisis General de Tipos de Variables ---
print("\n--- 11. Análisis General de Tipos de Variables ---")

categorical = []
numerical = []
for feature in df.columns:
    if df[feature].dtype == object:
        categorical.append(feature)
    else:
        numerical.append(feature)

print(f"\nVariables Categóricas: {categorical}")
print(f"Variables Numéricas: {numerical}")

print("\nHistogramas de todas las variables numéricas:")
df[numerical].hist(figsize=(20, 12), bins=100, color='lightgreen')
plt.suptitle("Histogramas de Todas las Variables Numéricas", y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Ajustar layout para el título superior
plt.show()

print("\nEstadísticas descriptivas para variables categóricas:")
print(df.describe(include=['object']))

print("\nConteo de valores normalizado para todas las variables categóricas:")
fig, axes = plt.subplots(ncols=4, nrows=3, figsize=(24, 18)) # Ajustar nrows/ncols si hay más de 12 categóricas
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6) # Aumentar hspace

for i, feature in enumerate(categorical):
    row = i // 4
    col = i % 4
    df[feature].value_counts(normalize=True).plot(kind='bar', ax=axes[row, col], color='lightgreen')
    axes[row, col].set_title(feature)
    axes[row, col].tick_params(axis='x', rotation=45) # Rotar etiquetas x
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

# --- 12. Análisis de Correlación General ---
print("\n--- 12. Análisis de Correlación General ---")

print("\nTabla de correlación de todas las variables numéricas:")
correlation_table = df[numerical].corr()
print(correlation_table)

print("\nMapa de calor de la tabla de correlación:")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_table, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación de Variables Numéricas")
plt.tight_layout()
plt.show()

print("\nGráficos de dispersión de variables numéricas vs variable objetivo 'y':")
# Asegurarse de que el número de subplots sea adecuado para todas las variables numéricas
num_cols_per_row = 4
num_rows = (len(numerical) + num_cols_per_row - 1) // num_cols_per_row # Calcular filas necesarias
fig, axes = plt.subplots(ncols=num_cols_per_row, nrows=num_rows, figsize=(num_cols_per_row * 6, num_rows * 5))
plt.subplots_adjust(wspace=0.3, hspace=0.4) # Ajustar espaciado

# Aplanar `axes` para iterar fácilmente si nrows > 1
axes_flat = axes.flatten() if num_rows > 1 else axes

for i, feature in enumerate(numerical):
    if feature == 'y': # Skip 'y' as it's the target
        continue
    ax = axes_flat[i]
    df.plot(x=feature, y='y', kind='scatter', ax=ax, color='green', alpha=0.6)
    ax.set_title(f"{feature} vs Suscripción")
    ax.set_xlabel(feature)
    ax.set_ylabel("Suscripción (y)")
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()


# --- 13. Análisis con Funciones Adicionales (Top N) ---
print("\n--- 13. Análisis con Funciones Adicionales (Top N) ---")

print("\nBoxplot de edad por las 3 educaciones principales:")
top_3_education = df.education.value_counts().sort_values(ascending=False).head(3).index.values
sns.boxplot(y="education", x="age", data=df[df.education.isin(top_3_education)], orient="h")
plt.title("Distribución de Edad por Educación (Top 3)")
plt.xlabel("Edad")
plt.ylabel("Nivel de Educación")
plt.tight_layout()
plt.show()

print("\n--- Análisis de Datos Bancarios Completado ---")