from utils import h2, init_connection
import utils as utils
import streamlit as st
import snowflake.snowpark.functions as F
import plotly.express as px
import constants as constant

# Configuración sobre la página
st.set_page_config(page_title='Fraud detection', layout='wide', initial_sidebar_state='auto')


session = init_connection()

# Titulo y subtitulos
st.markdown(constant.title, unsafe_allow_html=True)
st.markdown(constant.subtitle, unsafe_allow_html=True)

# Pestañas para dividir el contenido
tab1, tab2, tab3 = st.tabs(["EDA", "ML Training", "Inference"])

# Pestaña para el EDA básico
with tab1:

    # Dataframe de Snowpark
    dataframe = session.table(constant.table)
    dataframe = dataframe.limit(10000)

    # Dataframe de Pandas
    data = dataframe.toPandas()
    # Nombres de las columnas a minúsculas
    data.columns = data.columns.str.lower()

    # Columnas para colocar la información dentro de la app
    left_col, right_col = st.columns([5, 5])

    with left_col:
        h2("Amount of transactions")
        amount_transactions_data = utils.get_amount_by_type(data)

        # Usamos gráfico de plotly para mostrar los valores en forma de gráfico de tarta
        fig = px.pie(amount_transactions_data, values="count", names="type")
        st.write(fig)

    with right_col:
        h2('Money consider in transactions')
        money_transactions_data = utils.get_money_by_type_agg(data)

        # Mostramos la información en forma de gráfico de barras
        st.bar_chart(money_transactions_data)

# Pestaña para el entrenamiento customizado del modelo RF
with tab2:

    st.subheader("Hyperparameters Tuning")

    # Columnas para colocar los hiperparámetros que se pueden seleccionar
    left_col, center_col, right_col = st.columns([5, 5, 5])

    with left_col:
        # Slider
        test_size_value = st.slider('Test size:',
                  min_value=0.1,
                  max_value=0.9
                  )

        # Cuadro de inserción de valor numérico con valores mínimos y máximos permitidos
        n_estimators_value = st.number_input(label="Number estimators: ", min_value=20, max_value=500)
        # Seleccionable con distintos valores
        criterion_value = st.selectbox(label="Criterion: ",
                                       options=constant.criterion_options,
                                       help=constant.criterion_help)
    with center_col:
        # Checkbox
        checkbox_depth = st.checkbox(label="Max Depth by default", value=True)

        # Entrada de número habilitado o no según si se ha marcado el checkbox
        max_depth_value = st.number_input(label="Max Depth", max_value=100, disabled=checkbox_depth)

        if checkbox_depth:
            max_depth_value = None

        min_samples_value = st.number_input(label="Min Samples Split", min_value=2, max_value=10)
        min_samples_leaf_value = st.number_input(label="Min Samples Leaf", min_value=1, max_value=10)

    with right_col:
        # Checkbox
        checkbox_bootstrap = st.checkbox(label="Bootstrap", value=True, help=constant.bootstrap_help)

        # Entrada de número habilitado o no según si se ha marcado el checkbox
        max_samples_value = st.number_input(label="Max Depth", max_value=100, disabled=checkbox_bootstrap,
                                            help=constant.max_depth_help)
        if checkbox_bootstrap:
            max_samples_value = None


    if st.button(label="Let's train!"):
        from machine_learning import train_rf_prediction_model
        from snowflake.snowpark import types as T
        from machine_learning import create_predict_udf

        # Entrenamiento del modelo con los hiperparámetros seleccionados
        model_trained_output = train_rf_prediction_model(session, constant.table, 'ISFRAUD', test_size_value,
                                             n_estimators_value, criterion_value, max_depth_value, max_samples_value)

        st.write(model_trained_output)

        # Creación de Procedimiento Almacenado para entrenar el modelo en Snowflake
        session.add_import("fraud_detection_app/machine_learning.py")
        session.add_import("fraud_detection_app/constants.py")

        # Registramos la función de entrenamiento a través de un procedimiento almacenado
        session.sproc.register(func=train_rf_prediction_model, name="train_rf_prediction_model", packages=["snowflake-snowpark-python"],
                               return_type=T.MapType(), replace=True)

        # Llamamos a la función que registra la UDF que podrá ser llamada para inferir
        create_predict_udf(session)

with tab3:

    # Fraud movements es una selección de operaciones solamente fraudulentas
    fraud_movements = session.table('fraud_movements')
    st.write(fraud_movements.select(*constant.features).toPandas())

    if st.button(label="Let's Inference!"):

        # Ejemplo de uso
        prediction = fraud_movements\
            .select(*constant.features, F.call_udf('predict_fraud_udf',
                  F.array_construct(F.col('TYPE_CASH_IN'), F.col('TYPE_CASH_OUT'), F.col('TYPE_DEBIT'),
                                    F.col('TYPE_PAYMENT'), F.col('TYPE_TRANSFER'), F.col('AMOUNT'),
                                    F.col('OLDBALANCEORG'), F.col('NEWBALANCEORIG'), F.col('OLDBALANCEDEST'),
                                    F.col('NEWBALANCEDEST'))).alias('PREDICTION')).toPandas()

        st.write(prediction)
