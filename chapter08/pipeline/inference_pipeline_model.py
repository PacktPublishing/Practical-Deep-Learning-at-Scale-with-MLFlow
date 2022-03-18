import click
import mlflow
import logging
from pathlib import Path
import os
from mlflow.models import ModelSignature
import pandas as pd
import json
from cachetools import LRUCache
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

CONDA_ENV = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, "conda.yaml")
MODEL_ARTIFACT_PATH = 'inference_pipeline_model'

class InferencePipeline(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import gcld3
        self.detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                        max_num_bytes=1000)
        self.finetuned_text_classifier = mlflow.pytorch.load_model(self.finetuned_model_uri)
    
    def __init__(self, finetuned_model_uri, inference_pipeline_uri=None):
        self.cache = LRUCache(100)
        self.finetuned_model_uri = finetuned_model_uri
        self.inference_pipeline_uri = inference_pipeline_uri

    def preprocessing_step_lang_detect(self, row):  
        language_detected = self.detector.FindLanguage(text=row[0])
        if language_detected.language != 'en':
            print("found Non-English language text.")
        return language_detected.language
    
    def preprocessing_step_cache(self, row):
        if row[0] in self.cache:
            print("found cached result")
            return self.cache[row[0]]

    # for a single row
    def sentiment_classifier(self, row):
        # preprocessing: check cache
        cached_response = self.preprocessing_step_cache(row)
        if cached_response is not None:
            return cached_response

        # preprocessing: language detection
        language_detected = self.preprocessing_step_lang_detect(row)

        # model inference
        pred_label = self.finetuned_text_classifier.predict({row[0]})

        # postprocessing: add additional metadata
        response = json.dumps({
                'response': {
                    'prediction_label': pred_label
                },
                'metadata': {
                    'language_detected': language_detected,
                },
                'model_metadata': {
                    'finetuned_model_uri': self.finetuned_model_uri,
                    'inference_pipeline_model_uri': self.inference_pipeline_uri
                },
            })

        # postprocessing: store response and input pair to cache
        self.cache[row[0]]=response

        return response
    
    def predict(self, context, model_input):
        results =model_input.apply(self.sentiment_classifier, axis=1,  result_type='broadcast')
        return results


# Input and Output formats
input = json.dumps([{'name': 'text', 'type': 'string'}])
output = json.dumps([{'name': 'text', 'type': 'string'}])
# Load model from spec
signature = ModelSignature.from_dict({'inputs': input, 'outputs': output})

@click.command(help="This program creates a multi-step inference pipeline model .")
@click.option("--finetuned_model_run_id", default=None,
              help="This is the mlflow run id for the finetuned_model")
@click.option("--pipeline_run_name", default="inference_pipeline_model_logging", help="This is the mlflow run name.")
def task(finetuned_model_run_id, pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        finetuned_model_uri = f'runs:/{finetuned_model_run_id}/model'
        inference_pipeline_uri = f'runs:/{mlrun.info.run_id}/{MODEL_ARTIFACT_PATH}'

        try:
            mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH, 
                                conda_env=CONDA_ENV, 
                                python_model=InferencePipeline(finetuned_model_uri, inference_pipeline_uri), 
                                signature=signature,
                                registered_model_name=MODEL_ARTIFACT_PATH
                            )
        except Exception as e:
            logger.error(e)
            mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH, 
                                conda_env=CONDA_ENV, 
                                python_model=InferencePipeline(finetuned_model_uri, inference_pipeline_uri), 
                                signature=signature
                            )
            pass
    
        logger.info("finetuned model uri is: %s", finetuned_model_uri)
        logger.info("inference_pipeline_uri is: %s", inference_pipeline_uri)
        mlflow.log_param("finetuned_model_uri", finetuned_model_uri)
        mlflow.log_param("inference_pipeline_uri", inference_pipeline_uri)
        mlflow.set_tag('pipeline_step', __file__)

    logger.info("finished logging inference pipeline model")


if __name__ == '__main__':
    task()
