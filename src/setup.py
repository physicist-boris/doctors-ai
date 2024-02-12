'''A fancy docstring'''
from setuptools import setup


setup(
   name='doctors-ai-pipeline-training',
   version='1.0',
   description='A useful module for automated pipeline',
   packages=['doctors_ai', "doctors_ai/tools"],  #same as name
   install_requires=['prefect', 'mlflow', 'joblib', "deepchecks", "scikit-learn",
                     "hyperopt", "minio"], #external packages as dependencies
)