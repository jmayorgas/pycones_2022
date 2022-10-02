import pandas as pd
import streamlit as st
from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F
import matplotlib.pyplot as plt
import constants as constant
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# Inicialización de la conexión
@st.experimental_singleton
def init_connection():
    return Session.builder.configs(st.secrets["snowflake"]).create()

def h2(content):
    st.markdown(
        f"<h2 style='text-align: center;'>{content}</h2>",
        unsafe_allow_html=True
    )

def h3(content):
    st.markdown(
        f"<h3 style='text-align: center;'>{content}</h3>",
        unsafe_allow_html=True
    )

def get_boroughs():
    session = init_connection()
    boroughs = session.sql(constant.distinct_borough_query).toPandas()
    boroughs = boroughs["BOROUGH"].values.tolist()
    boroughs = ['ALL'] + boroughs
    return tuple(boroughs)

def get_contributing_factors():
    session = init_connection()
    contributing_factors = session.sql(constant.distinct_contributing_factors_query).toPandas()
    contributing_factors = contributing_factors["CONTRIBUTING_FACTOR_VEHICLE_1"].values.tolist()
    contributing_factors = ['ALL'] + contributing_factors
    return tuple(contributing_factors)

def get_vehicles_types():
    session = init_connection()
    vehicle_types = session.sql(constant.distinct_vehicle_type_query).toPandas()
    vehicle_types = vehicle_types["VEHICLE_TYPE_CODE_1"].values.tolist()
    vehicle_types = ['ALL'] + vehicle_types
    return tuple(vehicle_types)

def get_contribution_factors():
    session = init_connection()
    contributing_factors = session.sql(constant.distinct_contributing_factors_query).toPandas()
    contributing_factors = contributing_factors["CONTRIBUTING_FACTOR_VEHICLE_1"].values.tolist()
    contributing_factors = ['ALL'] + contributing_factors
    return tuple(contributing_factors)

def build_query(date_from, date_to, borough = "ALL", contributing_factor = "ALL", vehicle_type="ALL"):
    query = constant.base_query

    where_clause = f""" WHERE CRASH_DATE >= '{str(date_from)}' AND
                              CRASH_DATE <= '{str(date_to)}' """
    query += where_clause

    if borough != 'ALL':
        query += f" AND BOROUGH = '{str(borough)}'"
    if contributing_factor != 'ALL':
        query += f" AND CONTRIBUTING_FACTOR_VEHICLE_1 = '{str(contributing_factor)}'"
    if vehicle_type != 'ALL':
        query += f" AND VEHICLE_TYPE_CODE_1 = '{str(vehicle_type)}'"

    query += ' LIMIT 4000'

    return query

def get_map_data(dataframe):
    map_data = dataframe[dataframe.latitude.notnull()]
    map_data = map_data[map_data['latitude'] != 0]
    return map_data

def compute_persons_injured(dataframe):
    return dataframe['number_of_persons_injured'].sum()

def compute_pedestrians_injured(dataframe):
    return dataframe['number_of_pedestrians_injured'].sum()

def compute_cyclist_injured(dataframe):
    return dataframe['number_of_cyclist_injured'].sum()

def compute_motorist_injured(dataframe):
    return dataframe['number_of_motorist_injured'].sum()

def get_crash_by_zip_code(dataframe):
    crash_by_zip_code = dataframe\
        .filter(~F.col('zip_code').isNull())\
        .groupBy('zip_code').count()\
        .sort(F.col('COUNT').desc())

    return crash_by_zip_code.toPandas()

def compute_temporary_features(dataframe):
    dataframe[['hour', 'minute', 'sec']] = dataframe['crash_time'].astype(str).str.split(':', expand=True).astype(int)
    dataframe['crash_time'] = dataframe['crash_time'].astype(str)
    return dataframe

def compute_day_of_week_feature(dataframe):
    dataframe['crash_date'] = pd.to_datetime(dataframe['crash_date'])
    dataframe['day_of_week'] = dataframe['crash_date'].dt.day_name()
    return dataframe

def get_timeline_chart_vehicle_data(dataframe):
    data = dataframe.groupby(['borough', 'crash_date'])\
        .size()\
        .reset_index()
    data.columns = ['borough', 'crash_date', 'crashes']

    data_pivoted = pd.pivot_table(
        data=data,
        index='crash_date',
        columns='borough',
        values='crashes'
    )

    data_pivoted = data_pivoted.fillna(0)
    data_pivoted.columns = data_pivoted.columns.tolist()

    return data_pivoted

def get_timeline_hour_chart_vehicle_data(dataframe):
    hour_crashes = dataframe.groupby(['borough', 'hour'])\
        .size()\
        .reset_index()

    hour_crashes.columns = ['borough', 'hour', 'crashes']

    hour_crashes_pivoted = pd.pivot_table(
        data=hour_crashes,
        index='hour',
        columns='borough',
        values='crashes'
    )

    hour_crashes_pivoted = hour_crashes_pivoted.fillna(0)
    hour_crashes_pivoted.columns = hour_crashes_pivoted.columns.tolist()

    return hour_crashes_pivoted

def get_day_of_week_chart_data(dataframe):
    day_of_week_chart = dataframe.groupby(['day_of_week', 'borough'])\
        .size()\
        .reset_index()
    day_of_week_chart.columns = ['day_of_week', 'borough', 'crashes']

    day_of_week_chart_pivoted = pd.pivot_table(
        data=day_of_week_chart,
        index='day_of_week',
        columns='borough',
        values='crashes'
    )

    # Ordenamos los índices
    day_of_week_chart_pivoted = day_of_week_chart_pivoted.reindex(constant.days_of_week_ordered)
    day_of_week_chart_pivoted = day_of_week_chart_pivoted.fillna(0)
    return day_of_week_chart_pivoted

def get_bar_chart_vehicle_data(dataframe):
    dataframe = dataframe.groupby(['vehicle_type_code_1'])['vehicle_type_code_1'] \
        .count() \
        .sort_values(ascending=False) \
        .to_frame()
    dataframe.columns = ['Crashs']

    return dataframe

def get_bar_chart_borough_factor_data(dataframe):
    bar_chart_borough_factor_data = dataframe\
        .groupby(['borough', 'contributing_factor_vehicle_1'])\
        .size()\
        .reset_index()

    bar_chart_borough_factor_data.columns = ['borough', 'contributing_factor_vehicle_1', 'crashes']

    borough_factor_pivoted = pd.pivot_table(
        data=bar_chart_borough_factor_data,
        index='borough',
        columns='contributing_factor_vehicle_1',
        values='crashes'
    )

    borough_factor_pivoted = borough_factor_pivoted.fillna(0)
    borough_factor_pivoted.columns = borough_factor_pivoted.columns.tolist()

    # Para escalar entre 0 y 1 por fila y tener los resultados en porcentaje
    borough_factor_pivoted = borough_factor_pivoted.div(borough_factor_pivoted.sum(axis=1), axis=0)*100
    return borough_factor_pivoted

def center_metrics():
    return st.markdown('''
        <style>
        /*center metric label*/
        [data-testid="stMetricLabel"] > div:nth-child(1) {
            justify-content: center;
        }

        /*center metric value*/
        [data-testid="stMetricValue"] > div:nth-child(1) {
            justify-content: center;
        }
        </style>
        ''', unsafe_allow_html=True)

def run_kmeans_local(df, label_var_1, label_var_2, n_clusters=2):
    kmeans = KMeans(n_clusters, random_state=0).fit(df[[label_var_1, label_var_2]])

    fig, ax = plt.subplots()#figsize=(constant.alto_figura, constant.ancho_figura))

    ax.grid(False)
    ax.set_facecolor('#FFF')
    ax.spines[['left', 'bottom']].set_visible(True)
    ax.spines[['left', 'bottom']].set_color('#4a4a4a')
    ax.tick_params(labelcolor='#4a4a4a')
    ax.xaxis.label.set(color='#4a4a4a', fontsize=20)
    ax.yaxis.label.set(color='#4a4a4a', fontsize=20)
    #-------------------------------------------#

    # Create scatterplot
    ax = sns.scatterplot(
        ax=ax,
        x=df[label_var_1],
        y=df[label_var_2],
        hue=kmeans.labels_,
        palette=sns.color_palette('colorblind', n_colors=n_clusters),
        legend=None
    )

    # Annotate cluster centroids
    for ix, [Var1, Var2] in enumerate(kmeans.cluster_centers_):
        ax.scatter(Var1, Var2, s=100, c="#a8323e")
        ax.annotate(
            f"Cluster #{ix+1}",
            (Var1, Var2),
            fontsize=12,
            color="#a8323e",
            xytext=(Var1+5, Var2+3),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#a8323e", lw=2),
            ha="center",
            va="center"
        )

    # Compute Sihouette Score
    cluster_labels = kmeans.predict(df[[label_var_1, label_var_2]])
    score = silhouette_score(df[[label_var_1, label_var_2]], cluster_labels)

    return [fig, score]
