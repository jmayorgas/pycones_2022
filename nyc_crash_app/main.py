from utils import h2, h3, init_connection, center_metrics
import utils as utils
import streamlit as st
import datetime
import plotly.express as px
import constants as constant

# Hagamos que la página se ajuste al ancho
st.set_page_config(page_title='NYC Crash', layout='wide', initial_sidebar_state='auto')


# Establecemos la conexión con SF
session = init_connection()

# Titulo y subtitulos
st.markdown(constant.title, unsafe_allow_html=True)
st.markdown(constant.subtitle_up, unsafe_allow_html=True)
st.markdown(constant.subtitle_down, unsafe_allow_html=True)

# Sidebar
# Filtro para fechas de ocurrencia del accidente
date_from = st.sidebar.date_input(
    "From date",
    datetime.date(2021, 1, 1)
)
date_to = st.sidebar.date_input(
    "To date",
    datetime.date(2022, 1, 1)
)

# Pestañas para dividir el análisis de los datos
tab1, tab2, tab3 = st.tabs(["Geography", "Temporaly", "Vehicles"])

# Pestaña con información Geográfica
with tab1:
    # Columnas para colocar la información dentro de la app
    left_col, right_col = st.columns([5, 5])
    # En la columna izda ponemos un seleccionable con los distritos
    with left_col:
        borough = st.selectbox(
            'Borough', utils.get_boroughs()
        )
    # En la columna dcha ponemos un seleccionable con los factores
    with right_col:
        contributing_factor = st.selectbox(
            'Contributing factor', utils.get_contributing_factors(),  key="factor_1"
        )

    # Construimos la query para obtener los datos según los filtros activos
    query = utils.build_query(date_from=date_from, date_to=date_to, borough=borough,
                              contributing_factor=contributing_factor)
    dataframe = session.sql(query)

    # Dataframe de Pandas
    data = dataframe.toPandas()
    # Nombres de las columnas a minúsculas
    data.columns = data.columns.str.lower()

    #Filtramos los datos porque no puede haber nulos (LAT, LON) en el mapa
    map_data = utils.get_map_data(data)

    # Columnas para colocar la información dentro de la app
    left_col, right_col = st.columns([15, 5])

    with left_col:
        h2("Events map")
        # Mapa con los accidentes
        st.map(map_data)

    # MÉTRICAS
    with right_col:
        h3('People injured')

        center_metrics()
        persons_injured = utils.compute_persons_injured(data)
        pedestrians_injured = utils.compute_pedestrians_injured(data)
        cyclist_injured = utils.compute_cyclist_injured(data)
        motorist_injured = utils.compute_motorist_injured(data)
        total_injured = persons_injured + pedestrians_injured + cyclist_injured + motorist_injured

        st.metric(label="Persons", value=persons_injured)
        st.metric(label="Pedestrians", value=pedestrians_injured)
        st.metric(label="Cyclist", value=cyclist_injured)
        st.metric(label="Motorists", value=motorist_injured)
        st.markdown("""---""")
        st.metric(label="Total", value=total_injured)

    h3('Top ZIP Codes with more crashes')

    # Columnas para colocar la información en 3 columnas
    left_col, center_col, right_col = st.columns([5, 5, 5])

    crash_by_zip_code = utils.get_crash_by_zip_code(dataframe)

    with left_col:
        st.metric(label=f"#1 ({crash_by_zip_code.loc[0].at['ZIP_CODE']})",
                  value=f"{crash_by_zip_code.loc[0].at['COUNT']}")
    with center_col:
        st.metric(label=f"#2 ({crash_by_zip_code.loc[1].at['ZIP_CODE']})",
                  value=crash_by_zip_code.loc[1].at['COUNT'])
    with right_col:
        st.metric(label=f"#3 ({crash_by_zip_code.loc[2].at['ZIP_CODE']})",
                  value=crash_by_zip_code.loc[2].at['COUNT'])

# Pestaña con información temporal
with tab2:
    # Filtro de los factores de contribución
    contributing_factor = st.selectbox(
        'Contributing factor', utils.get_contributing_factors(), key="factor_2")

    query = utils.build_query(date_from=date_from, date_to=date_to, borough=borough,
                              contributing_factor=contributing_factor)
    dataframe = session.sql(query)

    # Dataframe de Snowpark
    dataframe = session.sql(query)
    # Dataframe de Pandas
    data = dataframe.toPandas()
    # Nombres de las columnas a minúsculas
    data.columns = data.columns.str.lower()

    data = utils.compute_temporary_features(data)

    ## Gráfico de lineas con el número de accidentes por día
    h3("Crashes by borough")
    timeline_chart_vehicle_data = utils.get_timeline_chart_vehicle_data(data)
    st.line_chart(timeline_chart_vehicle_data)

    ## Gráfico de lineas con el número de accidentes por hora
    h3("Crashes by hour")

    timeline_chart_hour_vehicle_data = utils.get_timeline_hour_chart_vehicle_data(data)
    st.line_chart(timeline_chart_hour_vehicle_data)

    h3("Crashes by Weekday (using Plotly)")

    # Se crea una nueva feature para determinar el día de la semana
    data = utils.compute_day_of_week_feature(data)
    day_of_week_chart_data = utils.get_day_of_week_chart_data(data)

    # Gráfico con plotly
    day_of_week_chart_data.reset_index(inplace=True)
    day_of_week_chart_data = day_of_week_chart_data.rename(columns={'index':'day_of_week'})
    fig = px.line(day_of_week_chart_data, x="day_of_week", y=day_of_week_chart_data.columns[1:], title="",
                  labels=dict(day_of_week="Weekday", value="Crashes", variable="Borough"))

    # Insertamos un gráfico de plotly
    st.plotly_chart(fig, use_container_width=True)

    # Pestaña con información sobre los vehículos
    with tab3:
        # Ponemos un seleccionable con los tipos de vehículos
        vehicle_type = st.selectbox(
            'Vehicle type', utils.get_vehicles_types(), key="vehicle_type")

        query = utils.build_query(date_from=date_from, date_to=date_to, vehicle_type=vehicle_type)

        # Dataframe de Snowpark
        dataframe = session.sql(query)
        # Dataframe de Pandas
        data = dataframe.toPandas()
        # Nombres de las columnas a minúsculas
        data.columns = data.columns.str.lower()

        ## Gráfico de barras con número de accidentes por tipo de vehículo
        h3("Top Contributing factor by borough")
        bar_chart_vehicle_data = utils.get_bar_chart_vehicle_data(data)

        st.write(bar_chart_vehicle_data)

        # Gráfico de barras
        st.bar_chart(bar_chart_vehicle_data.head(10))



