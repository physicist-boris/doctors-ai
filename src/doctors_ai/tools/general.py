import os
import warnings
warnings.filterwarnings("ignore")
import requests
import tempfile
from typing import Dict, Tuple
import joblib
import mlflow
import pandas as pd
import sklearn.pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from dataclasses import dataclass
from typing import List, Tuple
import json
from mlflow.tracking import MlflowClient
from mlflow.entities.run import Run
from mlflow.pyfunc import PyFuncModel

from doctors_ai.tools.data import Dataset
from sklearn.metrics import accuracy_score

from prefect import task
from minio import Minio
from doctors_ai.tools.definitions import MINIO_API_HOST, MINIO_SECRET_KEY, MINIO_ACCESS_KEY
from doctors_ai.tools.definitions import SERVER_API_URL
from doctors_ai.tools.data import load_prep_dataset_from_minio, load_raw_datasets_from_minio


def stop_mlflow_run(flow, flow_run, state):
    """Utilitary function to stop the current run after flow ends

    Note:
        Parameters are necessary, if not Prefect complains.
    """
    mlflow.end_run()


def make_mlflow_artifact_uri(experiment_id: str = None) -> str:
    """Generates the URI for the experiment

    Not that necessary.
    """
    import urllib.parse

    if experiment_id is None:
        run = mlflow.get_experiment()
        if run is not None:
            raise ValueError
        experiment_id = run.info.experiment_id

    return urllib.parse.urljoin(mlflow.get_tracking_uri(), f"/#/experiments/{experiment_id}")


def log_metrics(metrics: Dict[str, float]):
    """Logs metrics to mlflow

    Not intended to be a prefect flow since we don't
    want to setup all the MLFlow setup. So we assume
    the MLFlow artifact tracking is set correctly.
    """
    mlflow.log_metric("train_accuracy", metrics["train"])
    mlflow.log_metric("valid_accuracy", metrics["val"])
    mlflow.log_metric("test_accuracy", metrics["test"])
    # Or almost the same:
    # mlflow.log_metrics(metrics)



@task
def raw_data_extraction(training_bucket: str) -> pd.DataFrame:
    minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY, 
                         secret_key=MINIO_SECRET_KEY, secure=False)
    dataframe, _ = load_raw_datasets_from_minio(minio_client, training_bucket)
    #raw_data = pd.concat(dataframes, ignore_index=True)
    return dataframe

@task
def prep_data_construction(training_bucket: str) -> pd.DataFrame:
    minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY, 
                         secret_key=MINIO_SECRET_KEY, secure=False)
    dataset = load_prep_dataset_from_minio(minio_client, training_bucket)
    dataframe, ds_info = load_raw_datasets_from_minio(minio_client, training_bucket)
    return dataset, ds_info

@task
def upload_artefacts_to_minio(artefact_bucket: str, local_path: str, )-> pd.DataFrame:
    minio_client = Minio(MINIO_API_HOST, access_key=MINIO_ACCESS_KEY, 
                         secret_key=MINIO_SECRET_KEY, secure=False)
    minio_client.fput_object(bucket_name=artefact_bucket, object_name="ml_model.joblib", file_path=local_path)

@task
def build_pipeline(params: dict) -> sklearn.pipeline.Pipeline:
    columns_to_drop = ['date', 'age_at_ed_visit']
    columns_to_spline = ["age", "heart_rate", "saturation", "systolic_bp",
                     "diastolic_bp", "resp_rate", "temp", "charlson_score"]
    column_transformer = ColumnTransformer(
    transformers=[
        ('column_dropper', 'drop', columns_to_drop),
    ("splines", SplineTransformer(degree=params['n_degree'], n_knots=params['n_knots']),
                                      columns_to_spline)
    ], remainder = 'passthrough'
)

    pipe = make_pipeline(column_transformer, StandardScaler(), LogisticRegression())
    return pipe


@task
def fit_pipeline(data_transformer: sklearn.pipeline.Pipeline, dataset: Dataset):
    data_transformer.fit(dataset.train_x, dataset.train_y)
    return data_transformer




@task
def score(model, dataset) -> Dict[str, float]:
    """Returns the evaluation metrics"""
    model_eval = accuracy_evaluation(classifier=model, data=dataset)
    return model_eval


@task
def load_artifacts_from_mlflow(run):
    with tempfile.TemporaryDirectory() as d:
        best_classifier_obj = get_raw_artifacts_from_run(
            mlflow.get_tracking_uri(), run=run, tmp_dir_path=d
        )
        return joblib.load(best_classifier_obj)


@task
def deploy():
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
    }
    response = requests.post(SERVER_API_URL, headers=headers,json={'date': "2023-02-10"}, timeout=120)
    if response.status_code == 200:
        return response.json()
    return None


@dataclass
class Experiment:
    """
    A dataclass used to represent an Experiment on MLflow
    Attributes
    ----------
    tracking_server_uri : str
        the URI of MLFlow experiment tracking server
    name : str
        the name of the experiment
    """
    tracking_server_uri:str
    name:str


def get_best_run(experiment:Experiment, 
                 metric:str="valid_accuracy",
                 order:str="DESC",
                 filter_string:str="") -> Run:
    """Find the best experiment run entity

    Args:
        experiment (Experiment): experiment settings
        metric (str, optional): the metric for runs comparison. Defaults to "valid_accuracy".
        order (str, optional): the sorting order to find the best at first row w.r.t the metric. Defaults to "DESC".
        filter_string (str, optional): a string with which to filter the runs. Defaults to empty string, thus searching all runs.

    Returns:
        Run: the best run entity associated with the given experiment
    """
    best_runs = explore_best_runs(experiment, 1, metric, order, filter_string, False)
    return best_runs[0]

def explore_best_runs(experiment:Experiment, n_runs:int=5, metric:str="valid_accuracy", 
                      order:str="DESC", filter_string:str="", to_dataframe:bool=True) -> List[Run] | pd.DataFrame:
    """find the best runs from the given experiment

    Args:
        experiment (Experiment): Experiment settings
        n_runs (int, optional): the count of runs to return. Defaults to 5.
        metric (str, optional): the metric for runs comparison. Defaults to "valid_accuracy".
        order (str, optional): the sorting order w.r.t the metric to have the best at first row. Defaults to "DESC".
        filter_string (str, optional): a string with which to filter the runs. Defaults to empty string, thus searching all runs.
        to_dataframe (bool, optional): True for a derived Dataframe of Run ID / Perf. Metric. Defaults to True.

    Returns:
        List[Run] | pd.DataFrame: set of the best runs (Entity or Dataframe)
    """
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    client = MlflowClient(tracking_uri=experiment.tracking_server_uri)
    # Retrieve Experiment information
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    # Retrieve Runs information
    runs = client.search_runs(
        experiment_ids=experiment_id,
        max_results=n_runs,
        filter_string=filter_string,
        order_by=[f"metrics.{metric} {order}"]
    )
    if to_dataframe:
        run_ids = [run.info.run_id for run in runs if metric in run.data.metrics]
        run_metrics = [run.data.metrics[metric] for run in runs if metric in run.data.metrics]
        run_dataframe = pd.DataFrame({"Run ID": run_ids, "Perf.": run_metrics})
        return run_dataframe
    return runs

def get_raw_artifacts_from_run(tracking_server_uri, run, tmp_dir_path: os.PathLike='tmp') -> Tuple[str, str]:
    client = MlflowClient(tracking_uri=tracking_server_uri)
    # We assume that our saves will be under classifier/artifacts
    model_path = client.download_artifacts(run.info.run_id, 'classifier/artifacts/ml_model.joblib', tmp_dir_path)
    return model_path


class Registered_Model:

    def __init__(self, registry_uri, name, version=None, stage=None):
        assert version != None or stage != None, "You should specify either the version or stage"
        self.registry_uri = registry_uri
        self.name = name
        if version == None:
            self.version = get_model_version_by_stage(self.registry_uri, name, stage)
        else:
            self.version = version
        self.stage = stage
        
def load_production_model(tracking_uri:str, model_name:str) -> PyFuncModel:
    """
    Loads the model deployed in the 'Production' stage from the specified tracking URI.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to load.

    Returns:
    - PyFuncModel: The PyFuncModel representing the loaded production model.
    """
    return load_model_by_stage(tracking_uri, model_name, 'Production')
    
def load_staging_model(tracking_uri:str, model_name:str) -> PyFuncModel:
    """
    Loads the model deployed in the 'Staging' stage from the specified tracking URI.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to load.

    Returns:
    - PyFuncModel: The PyFuncModel representing the loaded staging model.
    """
    return load_model_by_stage(tracking_uri, model_name, 'Staging')

def get_latest_model_versions(tracking_uri:str, model_name:str) -> List[Dict]:
    """
    Retrieves the latest model versions and their stages for a specified model.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to retrieve latest versions for.

    Returns:
    - List[Dict]: A list of dictionaries containing version and stage information
                  for the latest versions of the specified model.
                  Example: [{"version": "1", "stage": "Production"}, ...]
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    latest_versions = client.get_latest_versions(name=model_name)
    return [{"version": version.version, "stage": version.current_stage}
            for version in latest_versions]

def transition_model_to_staging(tracking_uri:str, 
                                model_name:str, model_version:str) -> None:
    """
    Transitions a specific model version to the 'Staging' stage.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to transition.
    - model_version (str): The version of the model to transition to the 'Staging' stage.

    Note:
    - This function transitions the specified model version to the 'Staging' stage.
      It does not archive existing versions by default.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging",
        archive_existing_versions=False
    )

def transition_model_to_production(tracking_uri:str, 
                                   model_name:str, model_version:str) -> None:
    """
    Transitions a specific model version to the 'Production' stage.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to transition.
    - model_version (str): The version of the model to transition to the 'Production' stage.

    Note:
    - This function transitions the specified model version to the 'Production' stage.
      It archives existing versions by default.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
        archive_existing_versions=True
    )

def update_model_description(tracking_uri:str, model_name:str, 
                             model_version:str, description:str) -> None:
    """
    Updates the description of a specific model version.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to update.
    - model_version (str): The version of the model to update.
    - description (str): The new description for the model version.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.update_model_version(
        name=model_name,
        version=model_version,
        description=description
    )
        
def tag_model(tracking_uri:str, 
              model_name:str, model_version:str, tags:Dict) -> None:
    """
    Tags a specific model version with provided key-value pairs.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to tag.
    - model_version (str): The version of the model to tag.
    - tags (Dict): A dictionary containing key-value pairs to tag the model version.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    for tag_k, tag_v in tags.items():
        # Tag using model version
        client.set_model_version_tag(name=model_name, 
                                    version=f'{model_version}', 
                                    key=tag_k, value=tag_v)
    
def load_model_by_stage(tracking_uri:str, 
                        model_name:str, model_stage:str) -> PyFuncModel:
    """
    Loads a model based on its name and deployment stage.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to load.
    - model_stage (str): The deployment stage of the model ('Production', 'Staging', etc.).

    Returns:
    - PyFuncModel: The loaded model in the specified stage.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{model_stage}"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return loaded_model

def get_model_version_by_stage(tracking_uri, model_name, model_stage):
    latest_model_versions = get_latest_model_versions(tracking_uri, model_name)
    for model in latest_model_versions:
        if model['stage'] == model_stage:
            return model['version']
    return None

def load_model_by_version(tracking_uri:str, 
                          model_name:str, model_version:str) -> PyFuncModel:
    """
    Loads a model based on its name and version number.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - model_name (str): The name of the model to load.
    - model_version (str): The version number of the model.

    Returns:
    - PyFuncModel: The loaded model of the specified version.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{model_version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return loaded_model

def register_model_from_run(tracking_uri:str, run:Run, model_name:str) -> None:
    """
    Registers a model generated from an MLflow Run.

    Args:
    - tracking_uri (str): The URI where the MLflow tracking server is running.
    - run (Run): MLflow Run object containing information about the run.
    - model_name (str): The desired name for the registered model.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = build_model_uri_from_run(run)
    mlflow.register_model(model_uri=model_uri, 
                          name=model_name)

def build_model_uri_from_run(run:Run) -> str:
    """
    Builds the model URI from the MLflow Run object.

    Args:
    - run (Run): MLflow Run object containing information about the run.

    Returns:
    - str: The model URI constructed from the run information.
    """
    artifact_path = json.loads(run.data.tags['mlflow.log-model.history'])[0]["artifact_path"]
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
    return model_uri


def accuracy_evaluation(classifier, data: Dataset, decimals: int = 3
) -> Dict[str, float]:
    """Compute binary classification accuracy scores on training/validation/testing data splits
       by a given pair of data transformer and classifier.

    Args:
    ----
        data_transfomer (Pipeline): sklearn feature engineering pipeline
        classifier (abc.ABCMeta): sklearn classifier class
        data (Dataset): datasets (training/validation/test)
        decimals (int, optional): number decimal digits of precision. Defaults to 3.

    Returns:
    -------
        Dict[str, float]: (keys: splits names, values: accuracy scores)
    """
    train_accuracy = accuracy_score(data.train_y.values, classifier.predict(data.train_x))
    val_accuracy = accuracy_score(data.val_y.values, classifier.predict(data.val_x))
    test_accuracy = accuracy_score(data.test_y.values, classifier.predict(data.test_x))
    return {
        "train": round(train_accuracy, decimals),
        "val": round(val_accuracy, decimals),
        "test": round(test_accuracy, decimals),
    }


class SKLModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to encapsulate Scikit-learn models for training and inference.
    """

    def load_context(self, context):
        """
        Loads the serialized Scikit-learn transformer and classifier artifacts.

        This method is invoked when an MLflow model is loaded using pyfunc.load_model(),
        constructing the Python model instance.

        Args:
        - context: MLflow context containing the stored model artifacts.
        """
        import joblib
        self.loaded_classifier = joblib.load(context.artifacts["model_path"])
        
    def predict(self, context, model_input):
        """
        Generates predictions using the loaded Scikit-learn transformer and classifier.

        This method retrieves the Scikit-learn transformer and classifier artifacts.

        Args:
        - context: MLflow context containing the stored model artifacts.
        - model_input: Input data to be processed by the model.

        Returns:
        - Tuple: Loaded transformer and classifier artifacts.
        """
        return self.loaded_classifier