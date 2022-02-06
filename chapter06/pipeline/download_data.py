import click
import mlflow
from flash.core.data.utils import download_data
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@click.command(help="This program downloads data for finetuning a deep learning model for sentimental classification.")
@click.option("--download_url", default="https://pl-flash-data.s3.amazonaws.com/imdb.zip",
              help="This is the remote url where the data will be downloaded")
@click.option("--local_folder", default="./data", help="This is a local data folder.")
@click.option("--pipeline_run_name", default="chapter05", help="This is a mlflow run name.")
def task(download_url, local_folder, pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logger.info("Downloading data from  %s", download_url)
        download_data(download_url, local_folder)
        mlflow.log_param("download_url", download_url)
        mlflow.log_param("local_folder", local_folder)
        mlflow.log_param("mlflow run id", mlrun.info.run_id)
        mlflow.set_tag('pipeline_step', __file__)
        mlflow.log_artifacts(local_folder, artifact_path="data")

    logger.info("finished downloading data to %s", local_folder)


if __name__ == '__main__':
    task()
