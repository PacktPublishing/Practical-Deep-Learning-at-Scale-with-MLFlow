import logging
import os

import click
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download_data",
    "fine_tuning_model",
    "inference_pipeline_model"
]


@click.command()
@click.option("--pipeline_steps", default="all", type=str)
def run_pipeline(pipeline_steps):

    # Setup the mlflow experiment and AWS access for local execution environment
    # # if you run this project remotely, then comment out the following four lines
    # os.environ["MLFLOW_TRACKING_URI"] = "http://localhost"
    # os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    # os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    # os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    # set up mlflow experiment name
    # Note this can also be setup through the environment variable
    # export MLFLOW_EXPERIMENT_NAME=/Shared/dl_model_chapter08
    # for local execution mode, you can set it up without full path, i.e., dl_model_chapter08
    # for remote execution mode in Databricks, use the full path, i.e., /Shared/dl_model_chapter08
    EXPERIMENT_NAME = "dl_model_chapter08"
    #EXPERIMENT_NAME = "/Shared/dl_inference_model"
    # mlflow.create_experiment(EXPERIMENT_NAME, "s3://annotation-databricks-access-granted/ShortTerm/Users/yongliu/deploy")
    mlflow.set_experiment(EXPERIMENT_NAME)
    # experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    # logger.info("pipeline experiment_id: %s", experiment.experiment_id)

    # Steps to execute
    active_steps = pipeline_steps.split(",") if pipeline_steps != "all" else _steps
    logger.info("pipeline active steps to execute in this run: %s", active_steps)

    with mlflow.start_run(run_name='pipeline', nested=True) as active_run:
        if "download_data" in active_steps:
            download_run = mlflow.run(".", "download_data", parameters={})
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
            file_path_uri = download_run.data.params['local_folder']
            logger.info('downloaded data is located locally in folder: %s', file_path_uri)
            logger.info(download_run)

        if "fine_tuning_model" in active_steps:
            try:
                if file_path_uri is not None:
                    fine_tuning_run = mlflow.run(".", "fine_tuning_model", parameters={"data_path": file_path_uri})
            except Exception as e:
                logger.error('Use default data path value due to Error: %s', e)
                fine_tuning_run = mlflow.run(".", "fine_tuning_model")
                pass
            fine_tuning_run_id = fine_tuning_run.run_id
            fine_tuning_run = mlflow.tracking.MlflowClient().get_run(fine_tuning_run_id)
            logger.info(fine_tuning_run)
        
        if "inference_pipeline_model" in active_steps:
            if fine_tuning_run_id is not None and fine_tuning_run_id != 'None':
                inference_pipeline_model_run = mlflow.run(".", "inference_pipeline_model", parameters={"finetuned_model_run_id": fine_tuning_run_id})
                inference_pipeline_model_run = mlflow.tracking.MlflowClient().get_run(inference_pipeline_model_run.run_id)
                logger.info(inference_pipeline_model_run)
            else:
                logger.info("no finetuned model to create an inference pipeline model.")


    logger.info('finished mlflow pipeline run with a run_id = %s', active_run.info.run_id)


if __name__ == "__main__":
    run_pipeline()
