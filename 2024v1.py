import random
import openpyxl
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

ruta_archivo = r"2024.xlsx"

try:
    ruta_archivo = r"2024.xlsx"

    # Cargar el libro de trabajo
    libro_trabajo = openpyxl.load_workbook(ruta_archivo, data_only=True)

    # Seleccionar la hoja específica
    hoja = libro_trabajo["2024-1"]

    # Leer el DataFrame desde el archivo Excel
    dataframe = pd.DataFrame(hoja.values)

    # Renombrar las columnas según el encabezado de la primera fila
    dataframe.columns = dataframe.iloc[0]

    # Eliminar la primera fila, que ahora es el encabezado
    dataframe = dataframe.iloc[1:]

    # Filtrar las columnas ID, N, ODD
    columnas_interes = ["ID", "DATE", "N", "ODD", "WL", "OUT", "DIFF", "PERCENTAGE", "CONT", "SPORT", "EVENT"]
    dataframe = dataframe[columnas_interes]

    # Eliminar filas con valores nulos en la columna "N"
    dataframe = dataframe.dropna(subset=["N"])

    df_all = dataframe
    df_sports = dataframe
    df_sports_bets = dataframe
    # Acceder a los valores calculados en lugar de las fórmulas en la columna "OUT"
    #dataframe["OUT"] = hoja["OUT"].value

    # Cerrar el libro de trabajo
    libro_trabajo.close()

except FileNotFoundError:
    print(f"El archivo {ruta_archivo} no se encontró.")
except Exception as e:
    print(f"Ocurrió un error al abrir el archivo: {e}")





promedio_ODD = round(dataframe["ODD"].mean(), 3)
print(f"Promedio ODD: {promedio_ODD}")

ultimo_cero_idx = dataframe[dataframe["WL"].astype(float) == 0].index.max()
if pd.isnull(ultimo_cero_idx):
    suma_desde_ultimo_cero = dataframe["WL"].astype(float).sum()
else:
    suma_desde_ultimo_cero = dataframe.loc[ultimo_cero_idx:]["WL"].astype(float).sum()
print(f"Streak W: {suma_desde_ultimo_cero}")

total_valores_WL = len(dataframe["WL"])
resultado_division = round(dataframe["WL"].sum() / total_valores_WL, 3)
print(f"W %: {resultado_division}")

x1 = promedio_ODD
x2 = suma_desde_ultimo_cero
x3 = resultado_division



iterador = 0

formula_t0 = 0.9

while iterador < 10:
    a = random.random()
    b = random.random()
    c = random.random()
    formula_t1 = x1*a + x2*b*0.1 + x3*c

    if (formula_t1 < 1 and formula_t1 > formula_t0):
        print(f"F(x1, x2, x3) disminuyó de {formula_t0} a {formula_t1}")
        formula_t0 = formula_t1
        iterador = iterador + 1

print(f"f({a}, {b}, {c}) = {formula_t0}")
print(f"f({round(a, 3)}, {round(b, 3)}, {round(c, 3)}) "
      f"= {x1}*{round(a, 3)} + {x2}*{round(b, 3)} + {x3}*{round(c, 3)} "
      f"= {round(formula_t0, 6)}")

#cuota = input("Ingresar ODD: ")
cuota = 1.15

print(f"f({round(a, 3)}, {round(b, 3)}, {round(c, 3)}) "
      f"= {cuota}*{round(a, 3)} + 0.1*{x2}*{round(b, 3)} + {x3}*{round(c, 3)} "
      f"= {round(formula_t0, 6)}")
print(f"f({round(a, 3)}, {round(b, 3)}, {round(c, 3)}) = {round(a*float(cuota) + x2*b*0.1 + x3*c, 6)}")

formula_t2 = round(a*float(cuota) + x2*b*0.1 + x3*c, 6)

if (formula_t2 <= 1):
    print(f"BET WR: {formula_t2}%")
else:
    formula_t2 = 2 - formula_t2
    print(f"BET WR: {formula_t2*0.66}%")


def calcularStreak():
    fecha_maxima = dataframe["DATE"].max()
    dataframe_fecha_maxima = dataframe[dataframe["DATE"] == fecha_maxima]

    if (dataframe_fecha_maxima["WL"] == 0).any():
        ultimo_wl_igual_0_index = dataframe_fecha_maxima[dataframe_fecha_maxima["WL"] == 0].index.max()
        suma_wl_desde_0_en_fecha_maxima = dataframe_fecha_maxima.loc[ultimo_wl_igual_0_index:]["WL"].sum()
    else:
        suma_wl_desde_0_en_fecha_maxima = dataframe_fecha_maxima["WL"].sum()
    return suma_wl_desde_0_en_fecha_maxima

def calcularWR():
    porcentaje_wl_igual_1_total = (dataframe[dataframe["WL"] == 1].shape[0] / dataframe.shape[0]) * 100

    fecha_maxima = dataframe["DATE"].max()
    porcentaje_wl_igual_1_sin_fecha_maxima = (dataframe[
                                                  (dataframe["WL"] == 1) & (dataframe["DATE"] != fecha_maxima)].shape[
                                                  0] / dataframe[dataframe["DATE"] != fecha_maxima].shape[0]) * 100
    return round(porcentaje_wl_igual_1_total-porcentaje_wl_igual_1_sin_fecha_maxima, 2)




st.title("Locura 2024.")
st.subheader("Primer período desde 01-01-2024 hasta 30-06-2024.")

col1, col2, col3 = st.columns(3)
col1.metric(label="Streak de victorias", value=int(x2), delta=calcularStreak(), delta_color="normal")
col2.metric(label="Media de cuotas", value=x1)
col3.metric(label="Win rate", value=f"{round(x3 * 100, 3)} %", delta=f"{calcularWR()}%", delta_color="normal")



#st.subheader("Dataframe.")
#st.dataframe(dataframe)


#col1 = st.columns(1)[0]
#col1.metric(label="Precisión de predicción del resultado para la siguiente apuesta", value=f"{round(formula_t2*100, 3)} %", delta=2, delta_color="normal")



def graphWL(dataframe):
    st.subheader("Wins and losses")

    # Supongamos que ya tienes tu DataFrame con las columnas 'ODD' y 'WL'
    # Definir colores según los valores de 'WL'
    colores = dataframe['WL'].map({1: 'lightgreen', 0: 'salmon'})

    # Crear un gráfico de dispersión para 'WL' igual a 1
    fig, ax = plt.subplots()
    scatter_wl_1 = ax.scatter(dataframe[dataframe['WL'] == 1]['ODD'], dataframe[dataframe['WL'] == 1]['ODD'],
                              label='Wins', c='lightgreen', s=30)

    # Crear un gráfico de dispersión para 'WL' igual a 0
    scatter_wl_0 = ax.scatter(dataframe[dataframe['WL'] == 0]['ODD'], dataframe[dataframe['WL'] == 0]['ODD'],
                              label='Losses', c='salmon', s=30)

    # Configurar etiquetas y título
    ax.set_xlabel('ODD', fontsize=8)
    ax.set_ylabel('ODD', fontsize=8)

    # Agregar leyenda combinando ambos conjuntos de puntos
    ax.legend(handles=[scatter_wl_1, scatter_wl_0], fontsize=8)

    # Configurar tamaños de etiquetas
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



def graphSports(dataframe):
    st.subheader("Sports")

    dataframe_filtrado = dataframe[dataframe['SPORT'] != 'DPI']

    # Crear una figura y ejes
    fig, ax = plt.subplots()

    # Obtener colores únicos para cada valor único en la columna 'SPORT'
    colores_por_sport = {sport: f'C{i}' for i, sport in enumerate(dataframe_filtrado['SPORT'].unique())}

    # Iterar sobre cada valor único de 'SPORT' y dibujar los puntos correspondientes
    for sport, color in colores_por_sport.items():
        datos_sport = dataframe_filtrado[dataframe_filtrado['SPORT'] == sport]
        ax.scatter(datos_sport['ODD'], datos_sport['ODD'], label=sport, color=color, s=30)

    # Configurar etiquetas y título
    ax.set_xlabel('ODD', fontsize=8)
    ax.set_ylabel('ODD', fontsize=8)

    # Agregar leyenda
    ax.legend(fontsize=8)

    # Configurar límites de los ejes X e Y
    ax.set_xlim(dataframe_filtrado['ODD'].min() - 0.02, dataframe_filtrado['ODD'].max() + 0.02)
    ax.set_ylim(dataframe_filtrado['ODD'].min() - 0.02, dataframe_filtrado['ODD'].max() + 0.02)

    # Configurar tamaños de etiquetas
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

#graphSports(dataframe)


def graphLinearNonlinear(dataframe):
    st.subheader("Linear and nonlinear prediction")

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from matplotlib.ticker import FuncFormatter

    x = dataframe["N"].values.reshape(len(dataframe["N"]), 1)
    y = dataframe["PERCENTAGE"].values.reshape(len(dataframe["PERCENTAGE"]), 1)

    model = LinearRegression()
    model.fit(x, y)
    y_pred_linear = model.predict(x)

    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)
    model_nonlinear = LinearRegression()
    model_nonlinear.fit(x_poly, y)
    y_pred_nonlinear = model_nonlinear.predict(x_poly)

    df_resultados = pd.DataFrame({
        'x': x.flatten(),
        'y_PERCENTAGE': y.flatten(),
        'y_linear': y_pred_linear.flatten(),
        'y_nonlinear': y_pred_nonlinear.flatten()
    })

    # Visualizar los resultados en Streamlit
    st.write(df_resultados)

    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Real profit', color='lightgreen', s=30)
    ax.plot(x, y_pred_linear, label='Linear prediction', color='gray')
    ax.plot(x, y_pred_nonlinear, label='Second grade prediction', color='orange')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2%}'))
    ax.set_xlabel('# Bets').set_size(8)
    ax.set_ylabel('Profit').set_size(8)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    st.pyplot(fig)

graphLinearNonlinear(dataframe)


def graphND(dataframe):
    st.subheader("ODD's normal distribution")

    from scipy.stats import norm
    dataframe = dataframe.dropna(subset=['ODD'])

    # Crear los intervalos y contar la frecuencia en cada intervalo
    intervalos = np.arange(1.0, 2.1, 0.05)
    conteo_por_intervalo, bordes = np.histogram(dataframe['ODD'], bins=intervalos)

    fig, ax = plt.subplots()
    ax.bar(bordes[:-1], conteo_por_intervalo, width=0.05, align='edge', color='lightgreen', label="Odd's frequency")

    ax.set_xlabel('Odd').set_size(8)
    ax.set_ylabel('Frequency').set_size(8)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    # Agregar una curva de distribución normal
    mu, sigma = np.mean(dataframe['ODD']), np.std(dataframe['ODD'])
    x = np.linspace(1.0, 2.0, 100)
    pdf = norm.pdf(x, mu, sigma) * np.sum(conteo_por_intervalo) * (
                bordes[1] - bordes[0])  # Escalar la curva para que se ajuste al histograma
    ax.plot(x, pdf, color='orange', linestyle='--', label='Normal distribution (μ='
                                                          + str(round(mu, 2)) + ', σ='
                                                          + str(round(sigma, 2)) + ')')
    ax.legend(fontsize=8)

    st.pyplot(fig)

graphND(dataframe)








from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def graphRandomForests(dataframe):
    st.subheader("Features importance using random forests")
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
    dataframe['DAY'] = dataframe['DATE'].dt.strftime('%A')

    # Eliminar columnas 'DATE' y 'EVENT'
    dataframe = dataframe.drop(['DATE', 'ID', 'N', 'DIFF', 'PERCENTAGE', 'OUT'], axis=1)

    # Mapeo de valores únicos en 'CONT' a identificadores en valores enteros
    cont_mapping = {'ASIA': 1, 'EUROPE': 2, 'NA': 3, 'LA': 4, 'BCI': 5}
    dataframe['CONT'] = dataframe['CONT'].map(cont_mapping)

    # Mapeo de valores únicos en 'SPORT' a identificadores en valores enteros
    sport_mapping = {'ESPORTS': 1, 'TABLE TENNIS': 2, 'BASKETBALL': 3, 'DPI': 4, 'ENTERTAINMENT': 5}
    dataframe['SPORT'] = dataframe['SPORT'].map(sport_mapping)

    # Mapeo de valores únicos en 'DIA' a identificadores en valores enteros
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    dataframe['DAY'] = dataframe['DAY'].map(day_mapping)

    # Mapeo de valores únicos en 'EVENT' a identificadores en valores enteros
    dataframe['EVENT'] = dataframe['EVENT'].astype('category').cat.codes + 1

    # Asegurarse de que 'WL' sea una columna categórica si es multiclase
    dataframe['WL'] = dataframe['WL'].astype('category')

    # Separar las características (X) y la variable objetivo (y)
    X = dataframe.drop("WL", axis=1)
    y = dataframe["WL"]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Escalar las características para mejorar el rendimiento del modelo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crear el modelo RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, random_state=1)

    # Entrenar el modelo
    forest.fit(X_train_scaled, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = forest.predict(X_test_scaled)

    # Evaluar la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy}")

    # Obtener las etiquetas de las características y la importancia
    feat_labels = X.columns
    importances = forest.feature_importances_

    # Ordenar las características por importancia
    indices = np.argsort(importances)[::-1]

    colors = np.where(importances[indices] > 0.33, 'lightgreen', 'salmon')

    # Crear un gráfico de barras para visualizar la importancia de cada característica
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(X.shape[1]), importances[indices], align="center", color=colors)
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels(feat_labels[indices])
    ax.set_xlabel("Feature").set_size(13)
    ax.set_ylabel("Feature importance").set_size(13)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol='%'))

    # ax.set_title("Importancia de las Características en RandomForestClassifier")
    plt.tight_layout()

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



df_sports['WL'] = pd.to_numeric(df_sports['WL'])
winrate_sports = df_sports.groupby('SPORT')['WL'].mean()*100

num_apuestas = df_sports.groupby('SPORT')['N'].count()

df_sports = pd.DataFrame({
    'SPORT': winrate_sports.index,
    'WR': winrate_sports.values,
    'BETS': num_apuestas.values
})

print(df_sports)

import altair as alt

def graphWRSport(df_sports):
    st.subheader("SPORT's win rate")
    df_sports['WR'] = df_sports['WR'] / 100
    scatter_chart = alt.Chart(df_sports).mark_circle(size=90).encode(
        x='BETS',
        y=alt.Y('WR:Q', axis=alt.Axis(format='%')),
        color='SPORT',
        tooltip=['SPORT', 'BETS', alt.Tooltip('WR:Q', format='.1%')]
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(scatter_chart, use_container_width=True)

graphWRSport(df_sports)



def graphWinAlt(dataframe):
    st.subheader("ODD's dailies results")
    dataframe_filtrado = dataframe[dataframe['SPORT'] != 'DPI'].copy()
    dataframe_filtrado.loc[:, 'WIN OR LOSE'] = dataframe_filtrado['WL'].apply(lambda x: 'LOSS' if x == 0 else 'WIN')
    custom_colors = ['salmon', 'green']
    y_range = [0.95, dataframe_filtrado['ODD'].max()]
    scatter_chart = alt.Chart(dataframe_filtrado).mark_circle(size=100).encode(
        x='DATE:T',
        y=alt.Y('ODD:Q', scale=alt.Scale(domain=y_range)),
        color=alt.Color('WIN OR LOSE:N', scale=alt.Scale(range=custom_colors)),
        tooltip=['WIN OR LOSE', 'DATE', 'ODD']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(scatter_chart, use_container_width=True)

graphWinAlt(df_sports_bets)



def graphSportsAltDate(dataframe):
    st.subheader("SPORT's dailies results")
    dataframe_filtrado = dataframe[dataframe['SPORT'] != 'DPI']
    custom_colors = ['orange', 'green', 'gray', 'white']
    y_range = [0.95, dataframe_filtrado['ODD'].max()]
    scatter_chart = alt.Chart(dataframe_filtrado).mark_circle(size=100).encode(
        x='DATE:T',
        y=alt.Y('ODD:Q', scale=alt.Scale(domain=y_range)),
        color=alt.Color('SPORT:N', scale=alt.Scale(range=custom_colors)),
        tooltip=['SPORT', 'DATE', 'ODD']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(scatter_chart, use_container_width=True)

graphSportsAltDate(df_all)

from mlxtend.plotting import heatmap
import seaborn as sns


def graphCorrelacion(dataframe):
    st.subheader("Features correlation matrix")
    cm = np.corrcoef(dataframe.select_dtypes(include=np.number).values.T)
    fig, ax = plt.subplots()
    sns.heatmap(cm, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=dataframe.select_dtypes(include=np.number).columns,
                yticklabels=dataframe.select_dtypes(include=np.number).columns)
    st.pyplot(fig)

#graphCorrelacion(df_all)


graphRandomForests(df_all)


def graphCorrelacionv2(dataframe):
    st.subheader("Features correlation matrix")

    dataframe['DAY'] = dataframe['DATE'].dt.strftime('%A')
    dataframe['DAY'], _ = pd.factorize(dataframe['DAY'])

    dataframe['SPORT'], _ = pd.factorize(dataframe['SPORT'])
    dataframe['CONT'], _ = pd.factorize(dataframe['CONT'])
    dataframe['EVENT'], _ = pd.factorize(dataframe['EVENT'])
    dataframe['WL'] = pd.to_numeric(dataframe['WL'], errors='coerce')
    #dataframe['DIFF'] = pd.to_numeric(dataframe['DIFF'], errors='coerce')
    dataframe['ODD'] = pd.to_numeric(dataframe['ODD'], errors='coerce')

    numeric_columns = dataframe.select_dtypes(include=np.number)

    if len(numeric_columns.columns) >= 2:
        cm = np.corrcoef(numeric_columns.values.T)
        cm_df = pd.DataFrame(cm, columns=numeric_columns.columns, index=numeric_columns.columns)
        fig, ax = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.warning("No hay suficientes columnas numéricas para calcular la matriz de correlación.")

graphCorrelacionv2(df_all)





st.subheader("WIP")
col1, col2, col3 = st.columns(3)
col1.metric(label="Constante a", value=round(a, 4))
col2.metric(label="Constante b", value=round(b, 4))
col3.metric(label="Constante c", value=round(c, 4))

st.latex(r"P(x_1, x_2, x_3) = ax_1 + bx_2 + cx_3")


st.subheader("Clustering analysis")
from sklearn.datasets import make_blobs
#X, y = make_blobs(n_samples=250,n_features=2,centers=3,cluster_std=0.5,shuffle=True,random_state=0)
X = dataframe[['N', 'PERCENTAGE']].values
y = dataframe['PERCENTAGE'].values
fig, ax = plt.subplots()
ax.scatter(X[:, 0],
            X[:, 1],
            s=50,
            c='white',
            marker='o',
            edgecolor='black')
st.pyplot(fig)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='random',
            n_init = 10,
            max_iter = 300,
            tol = 1e-04,
            random_state = 0)
y_km = km.fit_predict(X)

fig, ax = plt.subplots()
ax.scatter(X[y_km == 0, 0],
           X[y_km == 0, 1],
           s=50, c='lightgreen',marker='s', edgecolor='black',label='Cluster 1')
ax.scatter(X[y_km == 1, 0],
           X[y_km == 1, 1],
           s=50, c='orange',marker='o', edgecolor='black',label='Cluster 2')
ax.scatter(X[y_km == 2, 0],
           X[y_km == 2, 1],
           s=50, c='mistyrose',marker='v', edgecolor='black',label='Cluster 3')
ax.scatter(km.cluster_centers_[:, 0],
           km.cluster_centers_[:, 1],
           s=250, c='red',marker='*', edgecolor='black',label='Centroids')
ax.grid()
ax.legend(scatterpoints=1)
st.pyplot(fig)
