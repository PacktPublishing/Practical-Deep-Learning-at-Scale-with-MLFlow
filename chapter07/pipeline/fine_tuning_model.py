import logging

import click
import flash
import mlflow
import torch
import torchmetrics
from flash.text import TextClassificationData, TextClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@click.command(help="This program finetunes a deep learning model for sentimental classification.")
@click.option("--foundation_model", default="prajjwal1/bert-tiny",
              help="This is the pretrained backbone foundation model")
@click.option("--fine_tuning_strategy", default="freeze", help="This is the finetuning strategy.")
@click.option("--data_path", default="data", help="This is the path to data.")
def task(foundation_model, fine_tuning_strategy, data_path):

    datamodule = TextClassificationData.from_csv(
        input_fields="review",
        target_fields="sentiment",
        train_file=f"{data_path}/imdb/train.csv",
        val_file=f"{data_path}/imdb/valid.csv",
        test_file=f"{data_path}/imdb/test.csv"
    )

    classifier_model = TextClassifier(backbone=foundation_model,
                                      num_classes=datamodule.num_classes, metrics=torchmetrics.F1(datamodule.num_classes))
    trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())

    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name="chapter07") as dl_model_tracking_run:
        trainer.finetune(classifier_model, datamodule=datamodule, strategy=fine_tuning_strategy)
        trainer.test()

        # mlflow log additional hyper-parameters used in this training
        mlflow.log_params(classifier_model.hparams)

        run_id = dl_model_tracking_run.info.run_id
        logger.info("run_id: {}; lifecycle_stage: {}".format(run_id,
                                                             mlflow.get_run(run_id).info.lifecycle_stage))
        mlflow.log_param("fine_tuning_mlflow_run_id", run_id)
        mlflow.set_tag('pipeline_step', __file__)


if __name__ == '__main__':
    task()
