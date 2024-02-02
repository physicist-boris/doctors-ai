'''A simple notebook demonstrating fitting and using a model on the data'''
from sqlite3 import connect
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer

DB_FILE = "../data/ed_visit.db"

con = connect(DB_FILE, check_same_thread=False)
df_train = pd.read_sql_query('SELECT * FROM ed_visit_train', con)
df_test = pd.read_sql_query('SELECT * FROM ed_visit_test', con)

columns = df_train.columns.to_list()
predictors = columns.copy()
PREDICTED = 'admitted'
predictors.remove(PREDICTED)

df_train_x = df_train[predictors]
df_train_y = df_train[PREDICTED]

df_test_x = df_test[predictors]
df_test_y = df_test[PREDICTED]

columns_to_drop = ['date', 'age_at_ed_visit']
columns_to_spline = ["age", "heart_rate", "saturation", "systolic_bp",
                     "diastolic_bp", "resp_rate", "temp", "charlson_score"]
column_transformer = ColumnTransformer(
    transformers=[
        ('column_dropper', 'drop', columns_to_drop),
        ("splines", SplineTransformer(degree=3, n_knots=5),
                                      columns_to_spline)
    ], remainder = 'passthrough'
)

pipe = make_pipeline(column_transformer, StandardScaler(), LogisticRegression())
fitted_train = pipe.fit(df_train_x, df_train_y)

predicted = pipe.predict(df_test_x)
predicted_probs = pipe.predict_proba(df_test_x)[:, 1]

truth = df_test_y

# Sum of probabilities is the most likely number of admissions
print(f"Predicted number admissions: {np.sum(predicted_probs)}" )
print(f"Actual number admissions: {np.sum(truth == 'yes')}" )

# # Accuracy on test
# np.sum(predicted == truth) / len(predicted)

# # AUC
# from sklearn import metrics
# fpr, tpr, thresholds = metrics.roc_curve(truth, predicted_probs, pos_label='yes')
# metrics.auc(fpr, tpr)
