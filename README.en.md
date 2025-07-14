[Versión en Español](README.md)

# Bank Data Visualization for Marketing 📊

This project primarily focuses on exploring and visualizing a dataset related to bank telemarketing campaigns. Through various data visualization techniques using libraries like matplotlib, seaborn, and plotly, the goal is to understand customer characteristics and the impact of campaigns. This involves identifying patterns and relationships between demographic and economic variables and the outcomes of term deposit subscriptions.

Visual analysis is fundamental for extracting initial insights and forming hypotheses before applying more complex Machine Learning models.

## Data Source 💾

This dataset is publicly available for research. Details are described in [Moro et al., 2014]. Please include this citation if you plan to use this database:

> [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

## Tecnologías Usadas 🐍
-   pandas: manipulación y análisis de datos tabulares, incluyendo la carga del conjunto de datos y transformaciones.
-   numpy: operaciones numéricas eficientes, especialmente con arreglos de datos.
-   matplotlib.pyplot: creación de gráficos estáticos y personalización básica de visualizaciones.
-   seaborn: creación de gráficos estadísticos atractivos e informativos, incluyendo distribuciones, relaciones bivariadas y mapas de calor.
-   plotly: creación de gráficos interactivos, que permiten una exploración más dinámica de los datos directamente en el navegador.
-   warnings: gestionar y filtrar advertencias.

## Installation Considerations ⚙️

To set up and run this project, using a conda environment is recommended. These libraries will help you create the necessary environment:

-  Install the main libraries (Python 3.9 or higher is recommended for compatibility with the latest library versions): 
    ```bash
    pip install pandas matplotlib seaborn numpy plotly
    ```
(Since plotly can have specific dependencies, pip is often more effective for its installation within Conda environments).

## Note ⚠️
The script includes !wget and !unzip commands to download and decompress the dataset. These commands are for Unix/Linux-based environments (like JupyterLab or Google Colab). If you run the script locally on Windows, you might need to download and decompress the bank-additional.zip file manually into the same directory where you save the script.

## Usage Example 📎

This code performs a series of steps for exploring and visualizing the bank telemarketing dataset:

1.  Data Download and Preparation: Loads the CSV into a pandas DataFrame and maps the target variable 'y' ('no'/'yes') to numerical values (0/1).
2.  Initial Data Exploration: Displays the first few rows of the DataFrame, column names, and the DataFrame's shape.
3.  Matplotlib Visualizations: Histograms for age distribution and bar charts for average age by marital status.
4.  Seaborn Visualizations:
    * `pairplot` to explore relationships between numerical variables.
    * `distplot` for variable distributions.
    * `jointplot` for bivariate relationships with scatter plots.
    * `boxplot` to compare age distributions by job type.
    * `heatmap` to visualize the correlation matrix or pivot tables.
5.  Interactive Plotly Visualizations: Interactive Box plots for age distribution by marital status.
6.  Segmented Analysis: Examples of visualizations for numerical (histogram, boxplot) and categorical (value count, countplot) variables.
7.  Comprehensive Variable Analysis: Classification of variables into numerical and categorical.
8.  Correlation Analysis: Calculation and visualization of the correlation matrix between numerical variables.
9.  Top N Analysis: Visualizing data for the N most frequent values of a categorical variable.

## Contributions 🖨️

If you're interested in contributing to this project or using it independently, consider:
-   Forking the repository.
-   Creating a new branch (git checkout -b feature/your-feature).
-   Making your changes and committing them (git commit -am 'Add new feature').
-   Pushing your changes to the branch (git push origin feature/your-feature).
-   Opening a 'Pull Request'.

## License 📜

This project is under the MIT License. Refer to the LICENSE file (if applicable) for more details.
