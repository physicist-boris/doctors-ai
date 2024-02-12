import logging
import warnings
warnings.filterwarnings("ignore")
import mlflow
from time import time
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
import prefect.context
import prefect.runtime.flow_run
from doctors_ai.tools.general import register_model_from_run
from doctors_ai.tools.general import Experiment as BankMarketingExperiment
from doctors_ai.tools.general import get_best_run
from doctors_ai.tools.general import get_model_version_by_stage
from doctors_ai.tools.general import transition_model_to_production
from doctors_ai.tools.general import transition_model_to_staging
from doctors_ai.tools.general import (
    build_pipeline,
    deploy,
    fit_pipeline,
    load_artifacts_from_mlflow,
    raw_data_extraction, 
    prep_data_construction,
    log_metrics,
    make_mlflow_artifact_uri,
    score,
    stop_mlflow_run, upload_artefacts_to_minio
)

from doctors_ai.tools.definitions import MLFLOW_TRACKING_URI
from prefect import flow, task
from prefect.artifacts import create_link_artifact
from prefect.logging import get_run_logger
from doctors_ai.tools.general import SKLModelWrapper
import os
import tempfile
import joblib
# N.B.: Note that we removed the feature_names
# N.B.: Note that due to how the hyperparameter tuning is done, we need to repeat
#       most steps.
def build_evaluation_func(data: Dataset, experiment_bag, ds_info: dict | None = None):
    """
    Create a new evaluation function
    :experiment_id: Experiment id for the training run
    :return: new evaluation function.
    """

    def eval_func(hparams):
        """
        Train sklearn model with given parameters by invoking MLflow run.
        :param params: Parameters to the train script we optimize over
        :return: The metric value evaluated on the validation data.
        """

        
        from mlflow.models.signature import infer_signature

        nonlocal ds_info
        if ds_info is None:
            ds_info = {}

        with mlflow.start_run(
            experiment_id=experiment_bag["mlflow_experiment_id"]
        ), tempfile.TemporaryDirectory() as temp_d:
            # Utility method to make things shorter
            tmp_fpath = lambda fpath: os.path.join(temp_d, fpath)

            # NOTE(Participant): This was added
            # N.B. We set a tag name so we can differentiate which Prefect run caused this
            #      Mlflow run. This will be useful to query models that were trained during
            #      this Prefect Run.
            mlflow.set_tag("prefect_run_name", experiment_bag["prefect_run_name"])

            # Params used to train RF
            (n_degree, n_knots) = hparams
            km_params = {"n_degree": n_degree, "n_knots": n_knots}
            # data_transformer = make_advanced_data_transformer(feature_names,
            #                                                   KMeans,
            #                                                   km_params)
            pipeline = build_pipeline.fn(km_params)
            # data_transformer.fit(data.train_x)
            fit_pipeline.fn(pipeline, dataset=data)

            joblib.dump(pipeline, tmp_fpath("ml_model.joblib"))
            # Log params
            mlflow.log_params({"n_degree": n_degree, "n_knots": n_knots})

            # Get Model Signature
            sample = data.train_x.sample(3)
            signature = infer_signature(data.train_x, pipeline.predict(data.train_x))
            # Log Model
            artifacts = {"model_path": tmp_fpath("ml_model.joblib")}
            mlflow_pyfunc_model_path = "classifier"
            mlflow.pyfunc.log_model(
                artifact_path=mlflow_pyfunc_model_path,
                python_model=SKLModelWrapper(),
                artifacts=artifacts,
                input_example=sample,
                signature=signature,
                extra_pip_requirements=["doctors_ai"],
            )
            
            # TODO(Participant): Reuse the method 'score' we defined above, but without the flow
            accuracy_dict = score.fn(model=pipeline, dataset=data)
            # Track metrics
            log_metrics(accuracy_dict)
            # Log the ds_info as a JSON file under the run's root artifact directory
            mlflow.log_dict(ds_info, "data.json")
            # Log the ds_info as a YAML file in a subdirectory of the run's root artifact directory
            mlflow.log_dict(ds_info, "data.yaml")
            return -accuracy_dict["val"]

    return eval_func


# N.B: We removed the feature_names
@task
def tune(data, max_runs, experiment_bag, ds_info):
    """
    Run hyperparameter optimization.
    """
    from hyperopt import fmin, hp, tpe
    from hyperopt.pyll import scope

    # Just a shortcut to both: 1) Set current experiment and 2) save the variable for experiment_id
    # For now, this is unused (the local functions will call set_experiment themselves)
    #experiment_id = mlflow.set_experiment(experiment_id=experiment_bag["mlflow_experiment_id"]).experiment_id
    # Search space for KMeans + RF
    space = [
        scope.int(hp.quniform("n_degree", 1, 5, q=1)),
        scope.int(hp.quniform("n_knots", 2, 8, q=1)),
    ]
    # Optimisation function that takes parent id and search params as input
    fmin(
        fn=build_evaluation_func(data, experiment_bag, ds_info=ds_info),
        space=space,
        algo=tpe.suggest,
        max_evals=max_runs,
    )


@flow(name="full-pipeline", on_completion=[stop_mlflow_run])
def automated_pipeline(
    data_bucket_for_training: str = "training-data-bucket",
    max_runs: int = 1,
    mlflow_experiment_name: str = "auto_pipeline",
):
    """Replaces experiments with HP tuning"""
    ######################################
    # Run setup
    ######################################
    # MLFlow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    unique_experiment_name = mlflow_experiment_name + "_" + str(int(time()))
    current_experiment = mlflow.set_experiment(experiment_name=unique_experiment_name)

    # Create a configuration object we can pass around
    pipeline_experiment_bag = {}
    pipeline_experiment_bag["mlflow_experiment_name"] = current_experiment.name
    pipeline_experiment_bag["mlflow_experiment_id"] = current_experiment.experiment_id
    pipeline_experiment_bag["prefect_run_name"] = prefect.runtime.flow_run.get_name()

    create_link_artifact(make_mlflow_artifact_uri(pipeline_experiment_bag["mlflow_experiment_id"]))

    # Logging setup
    run_logger = get_run_logger()
    # These are visible in the API Server
    run_logger.info("hiii")
    # These are visible in the worker
    logging.info(mlflow.get_tracking_uri())

    ######################################
    # Data Extraction
    ######################################
    df = raw_data_extraction(data_bucket_for_training)
    ######################################
    # Data Validation
    ######################################
    ds = Dataset(df, label= 'admitted')
    integ_suite = data_integrity()
    suite_result = integ_suite.run(ds)
    if not suite_result.passed():
        run_logger.warning('Failed data validation. See artifacts or GX UI for more details.')

    ######################################
    # Data preparation
    ######################################
    dataset, ds_info = prep_data_construction(data_bucket_for_training)

    ######################################
    # Training with hyperparameter search
    #####################################
    tune(
        data=dataset,
        max_runs=max_runs,
        experiment_bag=pipeline_experiment_bag,
        ds_info=ds_info,
    )
    
    current_experiment = BankMarketingExperiment(
        tracking_server_uri=mlflow.get_tracking_uri(),
        name=pipeline_experiment_bag["mlflow_experiment_name"],
    )
    
    # Get the best run of the current experiment of the current Prefect Run
    # (we used a tag in MLFlow. We set the tag key to "prefect_run_name")
    best_run = get_best_run(
        experiment=current_experiment,
        filter_string="tags.prefect_run_name = '{}'".format(pipeline_experiment_bag["prefect_run_name"]),
    )

    ######################################
    # Scoring
    ######################################
    best_classifier_obj = load_artifacts_from_mlflow(run=best_run)
    metrics = score(model=best_classifier_obj, dataset=dataset)
    run_logger.info(metrics)

    
    ######################################
    # Model register
    ######################################
    save_model = True
    saved_model_name = "Logistic regression"
    if save_model:
        run_logger.info("Saving model named: %s", saved_model_name)
        register_model_from_run(current_experiment.tracking_server_uri, best_run, saved_model_name)
        model_version = get_model_version_by_stage(current_experiment.tracking_server_uri, saved_model_name, "None")
        transition_model_to_staging(current_experiment.tracking_server_uri, saved_model_name, model_version) 
    
    ######################################
    # Model validation
    ######################################
    """
    result = validate_model(dataset, best_feat_eng_obj, best_classifier_obj, best_run.info.run_id)
    run_logger.info(f" {len(result.get_passed_checks())} of Model tests are passed.")
    run_logger.info(f" {len(result.get_not_passed_checks())} of Model tests are failed.")
    run_logger.info(f" {len(result.get_not_ran_checks())} of Model tests are not runned.")
    if result.passed(fail_if_check_not_run=True, fail_if_warning=True):
        run_logger.info("The Model validation succeeds")
        tag_model(current_experiment.tracking_server_uri, saved_model_name, model_version, {"Model Tests": "PASSED"})
    else:
        run_logger.info("The Model validation fails")
        tag_model(current_experiment.tracking_server_uri, saved_model_name, model_version, {"Model Tests": "FAILED"})
    """
    ######################################
    # Model deployment
    ######################################
    should_deploy = True
    if should_deploy:
        transition_model_to_production(current_experiment.tracking_server_uri, saved_model_name, model_version)
        with tempfile.TemporaryDirectory() as temp_d:
            # Utility method to make things shorter
            tmp_fpath = lambda fpath: os.path.join(temp_d, fpath)
            joblib.dump(best_classifier_obj, tmp_fpath("ml_model.joblib"))
            upload_artefacts_to_minio(artefact_bucket="artefacts", local_path=tmp_fpath("ml_model.joblib"))
        model_info = deploy()
        run_logger.info(model_info)


automated_pipeline()
