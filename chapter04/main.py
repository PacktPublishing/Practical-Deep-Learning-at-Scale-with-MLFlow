import logging
import os

import click
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download_data",
    "fine_tuning_model",
    "register_model"
]


@click.command()
@click.option("--steps", default="all", type=str)
def run_pipeline(steps):

    # Setup the mlflow experiment and AWS access
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    EXPERIMENT_NAME = "dl_model_chapter04"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    logger.info("pipeline experiment_id: %s", experiment.experiment_id)

    # Steps to execute
    active_steps = steps.split(",") if steps != "all" else _steps
    logger.info("pipeline active steps to execute in this run: %s", active_steps)

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
        if "download_data" in active_steps:
            download_run = mlflow.run(".", "download_data", parameters={})
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
            file_path_uri = download_run.data.params['local_folder']
            logger.info('downloaded data is located locally in folder: %s', file_path_uri)
            logger.info(download_run)

        if "fine_tuning_model" in active_steps:
            fine_tuning_run = mlflow.run(".", "fine_tuning_model", parameters={"data_path": file_path_uri})
            fine_tuning_run_id = fine_tuning_run.run_id
            fine_tuning_run = mlflow.tracking.MlflowClient().get_run(fine_tuning_run_id)
            logger.info(fine_tuning_run)

        if "register_model" in active_steps:
            if fine_tuning_run_id is not None and fine_tuning_run_id != 'None':
                register_model_run = mlflow.run(".", "register_model", parameters={"mlflow_run_id": fine_tuning_run_id})
                register_model_run = mlflow.tracking.MlflowClient().get_run(register_model_run.run_id)
                logger.info(register_model_run)
            else:
                logger.info("no model to register since no trained model run id.")
    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)


if __name__ == "__main__":
    run_pipeline()
