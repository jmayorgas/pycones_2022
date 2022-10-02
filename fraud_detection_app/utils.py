import streamlit as st
from snowflake.snowpark.session import Session


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

def get_test_values(data, n_rows):
    from utils import init_connection
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder

    df = data.toPandas()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # One-Hot-Encoding a la columna CASH_OUT
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['TYPE']]))

    encoder_df.index = df.index
    encoder_df.columns = encoder.get_feature_names(['TYPE'])

    df = df.drop('TYPE', axis=1)
    df = pd.concat([encoder_df, df], axis=1)

    session = init_connection()

    return session.create_dataframe(df.head(n_rows))

def get_amount_by_type(dataframe):
    amount_transactions_data = dataframe.groupby(['type']).size().reset_index()
    amount_transactions_data.columns = ['type', 'count']

    return amount_transactions_data

def get_money_by_type_agg(dataframe):
    money_transactions_data = dataframe.groupby(['type'])['amount'].agg(['sum'])
    money_transactions_data.columns = ['Quantity']

    return money_transactions_data