import click
import mlflow
import logging
from pathlib import Path
import os
from mlflow.models import ModelSignature
import json
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

CONDA_ENV = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, "conda.yaml")
MODEL_ARTIFACT_PATH = 'nlp_sentiment_classifier_explainer'

class SentimentAnalysisExplainer(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import pipeline
        import shap

        self.explainer = shap.Explainer(pipeline('sentiment-analysis', return_all_scores=True))
    
    def sentiment_classifier_explanation(self, row):
        # one row but needs to be a list 
        # so that words won't be split into chars
        shap_values = self.explainer([row['text']])
        return [pickle.dumps(shap_values)]

    def predict(self, context, model_input):
        model_input[['shap_values']] = model_input.apply(self.sentiment_classifier_explanation, axis=1, result_type='expand')
        model_input.drop(['text'], axis=1, inplace=True)
        return model_input
    
# Input and Output formats
input = json.dumps([{'name': 'text', 'type': 'string'}])
output = json.dumps([{'name': 'shap_values', 'type': 'string'}])
# # Load model from spec
signature = ModelSignature.from_dict({'inputs': input, 'outputs': output})

@click.command(help="This program creates an nlp sentiment classifier explainer .")
@click.option("--pipeline_run_name", default="nlp_sentiment_classifier_explainer", help="This is the mlflow run name.")
def task(pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH, 
                                conda_env=CONDA_ENV, 
                                python_model=SentimentAnalysisExplainer(), 
                                signature=signature
                            )
    
        mlflow.set_tag('pipeline_step', __file__)
    run_id = mlrun.info.run_id
    logger.info(f"finished logging nlp sentiment classifier explainer run_id: {run_id}")


if __name__ == '__main__':
    task()
