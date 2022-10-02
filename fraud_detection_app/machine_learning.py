import constants as constants

# Entrenamiento del modelo
def train_rf_prediction_model(session, table, target_column, test_size, n_estimators, criterion, max_depth, bootstrap):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    import os
    from joblib import dump

    # Lectura de la tabla y selección de las columnas
    df = session.table(table).select(constants.origin_features).limit(5000).toPandas()

    # Uso de OHE para la columna TYPE que es de tipo categórico
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # One-Hot-Encoding a la columna CASH_OUT
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['TYPE']]))
    encoder_df.index = df.index
    encoder_df.columns = encoder.get_feature_names(['TYPE'])
    # Borramos la columna original
    df = df.drop('TYPE', axis=1)
    # Anexamos los resultados al DF original
    df = pd.concat([encoder_df, df], axis=1)

    # División del Dataframe en X e Y
    df_x = df.drop(target_column, axis=1)
    df_y = df[target_column]

    # Split en train y test
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size,
                                                        random_state=12345, stratify=df_y)

    # Creación del modelo con los hiperparámetros establecidos por la App
    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                   bootstrap=bootstrap, class_weight="balanced")
    # Train
    model.fit(x_train, y_train)

    # Prediction
    y_pred = model.predict(x_test)

    from sklearn import metrics
    # Cálculo de métricas
    test_performance = {"Accuracy": metrics.accuracy_score(y_test, y_pred),
                        "Recall:": metrics.recall_score(y_test, y_pred),
                        "Precision:": metrics.precision_score(y_test, y_pred),
                        "F1 Score:": metrics.f1_score(y_test, y_pred),}

    # Subimos el modelo entrenado al stage
    model_output_dir = "/tmp"
    model_file = os.path.join(model_output_dir, 'model.joblib')
    dump(model, model_file)
    session.file.put(model_file, f"{constants.stage}/models/", overwrite=True)

    return test_performance

# Función para inferir
def predict_fraud(bank_movement: list):
    import sys
    import pandas as pd
    from joblib import load

    # Obtenemos ruta donde el modelo está guardado
    import_dir = sys._xoptions[constants.IMPORT_DIRECTORY_NAME]
    model_file = import_dir + 'model.joblib.gz'

    # Carga del modelo
    model = load(model_file)

    # Creamos un Dataframe para ser usado como entrada en la función de predict
    df = pd.DataFrame([bank_movement], columns=constants.features)

    # Predicción
    prediction = model.predict(df)[0]

    return prediction

# Creación de UDF para inferir
def create_predict_udf(session):
    from snowflake.snowpark import types as T

    session.clear_imports()
    session.clear_packages()

    # Se añaden los imports de los archivos que van a ser
    # necesarios para inferencia en SF
    session.add_import(f'{constants.stage}/models/model.joblib.gz')
    session.add_import("fraud_detection_app/machine_learning.py")
    session.add_import("fraud_detection_app/constants.py")

    # Paquetes a ser incluidos en el entorno de Anaconda
    dep_packages = ["pandas==1.3.5", "scipy==1.7.1", "scikit-learn==1.0.2", "setuptools==58.0.4", "joblib"]

    # Registro de UDF para inferencia
    fraud_prediction = session.udf.register(predict_fraud,
                                             name='predict_fraud_udf',
                                             is_permanent=True,
                                             stage_location=constants.stage,
                                             packages=dep_packages,
                                             input_types=[T.ArrayType()],
                                             return_type=T.IntegerType(),
                                             replace=True)


    return fraud_prediction