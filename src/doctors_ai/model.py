'''A tiny API to serve the model results'''
from sqlite3 import connect
from joblib import load
import pandas as pd
import numpy as np
from minio import Minio
import os


def output(date:str, minio_api_host:str, minio_access_key:str, minio_secret_key:str) -> tuple[int, int]:
    """_summary_

    Args:
        date (str): The date of the request as a YYYY-MM-DD string
        db_location (str, optional): Points to an SQLite db. Defaults to 'data/ed_visit.db'.
        model_location (str, optional): Points to a .joblib file. Defaults to 'data/model.joblib'.

    Returns:
        (int, int): (Predicted number of admissions, Number in ED)
    """
    minio_client = Minio(minio_api_host, access_key=minio_access_key, 
                         secret_key=minio_secret_key, secure=False)
    minio_client.fget_object("inference", "ed_visit.db", "temp_data.db")
    minio_client.fget_object("artefacts", "ml_model.joblib", "temp_path_model.joblib")
    model = load("temp_path_model.joblib")
    os.remove("temp_path_model.joblib")
    con = connect("temp_data.db", check_same_thread=False)
    df_test = pd.read_sql_query(f"SELECT * FROM ed_visit_test where date = '{date}'", con)
    con.close()
    os.remove("temp_data.db")
    columns = df_test.columns.to_list()
    predictors = columns.copy()
    predicted = 'admitted'
    predictors.remove(predicted)
    df_test_x = df_test[predictors]
    df_test_y = df_test[predicted]
    predicted_probs = model.predict(df_test_x)
    truth = df_test_y
    # Sum of probabilities is the most likely number of admissions
    predicted_probs = [True if element == 'yes' else False for element in predicted_probs]
    predicted_number_admissions = np.sum(predicted_probs).astype(int)
    number_in_ed = len(truth)
    return int(predicted_number_admissions), int(number_in_ed)
