# Visualización de Datos Bancarios para Marketing 📊

Este proyecto tiene como objetivo principal la exploración y visualización de un conjunto de datos relacionado con campañas de telemarketing bancario. A través de diversas técnicas de visualización de datos utilizando librerías como `matplotlib`, `seaborn`, y `plotly`, se busca entender las características de los clientes y el impacto de las campañas, identificando patrones y relaciones entre las variables demográficas, económicas y los resultados de las suscripciones a depósitos a plazo.

El análisis visual es fundamental para extraer insights iniciales y formular hipótesis antes de aplicar modelos de Machine Learning más complejos.

## Fuente de Datos 💾

Este conjunto de datos está disponible públicamente para investigación. Los detalles se describen en [Moro et al., 2014]. Por favor, incluye esta cita si planeas usar esta base de datos:

> [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

## Tecnologías Usadas 🐍
-   pandas: manipulación y análisis de datos tabulares, incluyendo la carga del conjunto de datos y transformaciones.
-   numpy: operaciones numéricas eficientes, especialmente con arreglos de datos.
-   matplotlib.pyplot: creación de gráficos estáticos y personalización básica de visualizaciones.
-   seaborn: creación de gráficos estadísticos atractivos e informativos, incluyendo distribuciones, relaciones bivariadas y mapas de calor.
-   plotly: creación de gráficos interactivos, que permiten una exploración más dinámica de los datos directamente en el navegador.
-   warnings: gestionar y filtrar advertencias.

## Consideraciones en Instalación ⚙️

Para configurar y ejecutar este proyecto, se recomienda utilizar un entorno `conda`. Estas librerias te ayudarán a crear el entorno necesario:

-  Instala las librerías principales (Se recomienda Python 3.9 o superior para compatibilidad con las últimas versiones de las librerías): 
    ```bash
    pip install pandas matplotlib seaborn numpy plotly
    ```
(Dado que `plotly` puede tener dependencias específicas, `pip` suele ser más efectivo para su instalación en entornos Conda).

## Nota ## ⚠️
El script incluye comandos `!wget` y `!unzip` para descargar y descomprimir el conjunto de datos. Estos comandos son para entornos basados en Unix/Linux (como JupyterLab o Google Colab). Si ejecutas el script localmente en Windows, es posible que necesites descargar y descomprimir el archivo `bank-additional.zip` manualmente en el mismo directorio donde guardes el script.

## Ejemplo de Uso 📎

Este código realiza una serie de pasos para la exploración y visualización del conjunto de datos de telemarketing bancario:

1.  Descarga y Preparación de Datos: Carga el CSV en un DataFrame de `pandas` y mapea la variable objetivo 'y' (`'no'`/`'yes'`) a valores numéricos (0/1).
2.  Exploración Inicial de Datos: Muestra las primeras filas del DataFrame, los nombres de las columnas y la forma del DataFrame.
3.  Visualizaciones con Matplotlib: Histogramas para la distribución de edad y gráficos de barras para la edad promedio por estado civil.
4.  Visualizaciones con Seaborn:
    * `pairplot` para explorar relaciones entre variables numéricas.
    * `distplot` para distribuciones de variables.
    * `jointplot` para relaciones bivariadas con dispersión.
    * `boxplot` para comparar distribuciones de edad por tipo de trabajo.
    * `heatmap` para visualizar la matriz de correlación o tablas pivote.
5.  Visualizaciones Interactivas con Plotly: Gráficos 'Box plots' interactivos para la distribución de edad por estado civil.
6.  Análisis Segmentado: Ejemplos de visualización para variables numéricas (histograma, boxplot) y categóricas (conteo de valores, `countplot`).
7.  Análisis Completo de Variables: Clasificación de variables en numéricas y categóricas.
8.  Análisis de Correlación: Cálculo y visualización de la matriz de correlación entre variables numéricas.
9.  Análisis de Top N: visualizar datos para los N valores más frecuentes de una variable categórica.

## Contribuciones 🖨️

Si te interesa contribuir a este proyecto o usarlo independiente, considera:
-   Hacer un "fork" del repositorio.
-   Crear una nueva rama (`git checkout -b feature/su-caracteristica`).
-   Realizar tus cambios y "commiteelos" (`git commit -am 'Agrega nueva característica'`).
-   Subir los cambios a la rama (`git push origin feature/su-caracteristica`).
-   Abrir un "Pull Request".

## Licencia 📜

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE (si aplica) para más detalles.
