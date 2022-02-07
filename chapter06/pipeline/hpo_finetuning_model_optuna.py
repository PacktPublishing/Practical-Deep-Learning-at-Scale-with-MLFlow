import logging
import os
import flash
import mlflow
import torchmetrics
from flash.text import TextClassificationData, TextClassifier

from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.optuna import OptunaSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@mlflow_mixin
def finetuning_dl_model(config, data_dir=None, num_epochs=3, num_gpus=0):
    datamodule = TextClassificationData.from_csv(
        input_fields="review",
        target_fields="sentiment",
        train_file=f"{data_dir}/imdb/train.csv",
        val_file=f"{data_dir}/imdb/valid.csv",
        test_file=f"{data_dir}/imdb/test.csv",
        batch_size=config['batch_size']
    )


    classifier_model = TextClassifier(backbone=config['foundation_model'],
                                      learning_rate=config['lr'],
                                      optimizer=config['optimizer_type'],
                                      num_classes=datamodule.num_classes,
                                      metrics=torchmetrics.F1(datamodule.num_classes)
                                      )
    mlflow.pytorch.autolog()
    metrics = {"loss": "val_cross_entropy", "f1": "val_f1"}
    trainer = flash.Trainer(max_epochs=num_epochs,
                            gpus=num_gpus,
                            progress_bar_refresh_rate=0,
                            callbacks=[TuneReportCallback(metrics, on='validation_end')])
    
    trainer.finetune(classifier_model, datamodule=datamodule, strategy=config['finetuning_strategies'])
    mlflow.log_param('batch_size',config['batch_size'])
    mlflow.set_tag('pipeline_step', __file__)


def run_hpo_dl_model(num_samples=10,
                     num_epochs=3,
                     gpus_per_trial=0,
                     tracking_uri=None,
                     experiment_name="hpo-tuning-chapter06"):

    data_dir = os.path.join(os.getcwd(), "data")

    # Set the MLflow experiment, or create it if it does not exist.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # define search algo and scheduler
    searcher = OptunaSearch()
    searcher = ConcurrencyLimiter(searcher, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler()

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "foundation_model": "prajjwal1/bert-tiny",
        "finetuning_strategies": "freeze",
        "optimizer_type": "Adam",
        "mlflow": {
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri()
        },
    }

    trainable = tune.with_parameters(
        finetuning_dl_model,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial)

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="f1",
        mode="max",
        config=config,
        num_samples=num_samples,
        search_alg=searcher,
        scheduler=scheduler,
        name="hpo_tuning_dl_model")
    
    logger.info("Best hyperparameters found were: %s", analysis.best_config)


def task():
    run_hpo_dl_model(num_samples=10,
                     num_epochs=3,
                     gpus_per_trial=0,
                     tracking_uri="http://localhost",
                     experiment_name="hpo-tuning-chapter06")


if __name__ == '__main__':
    task()
