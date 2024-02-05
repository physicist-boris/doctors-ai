'''A tiny API to serve the model results'''
from sqlite3 import connect
from joblib import load
import pandas as pd
import numpy as np


def output(date:str, db_location:str = 'data/ed_visit.db',
           model_location:str = 'data/model.joblib') -> tuple[int, int]:
    """_summary_

    Args:
        date (str): The date of the request as a YYYY-MM-DD string
        db_location (str, optional): Points to an SQLite db. Defaults to 'data/ed_visit.db'.
        model_location (str, optional): Points to a .joblib file. Defaults to 'data/model.joblib'.

    Returns:
        (int, int): (Predicted number of admissions, Number in ED)
    """
    model = load(model_location)
    con = connect(db_location, check_same_thread=False)
    df_test = pd.read_sql_query(f"SELECT * FROM ed_visit_test where date = '{date}'", con)
    columns = df_test.columns.to_list()
    predictors = columns.copy()
    predicted = 'admitted'
    predictors.remove(predicted)
    df_test_x = df_test[predictors]
    df_test_y = df_test[predicted]
    predicted_probs = model.predict_proba(df_test_x)[:, 1]
    truth = df_test_y
    # Sum of probabilities is the most likely number of admissions
    predicted_number_admissions = np.rint(np.sum(predicted_probs)).astype(int)
    number_in_ed = len(truth)
    return int(predicted_number_admissions), int(number_in_ed)
