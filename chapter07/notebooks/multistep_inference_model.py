# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import mlflow
from mlflow.models import ModelSignature
import pandas as pd
import json
from cachetools import LRUCache

# %%
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"


# %%

EXPERIMENT_NAME = "dl_model_chapter07"
mlflow.set_tracking_uri('http://localhost')
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)


# %%
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


# %%
# Input and Output formats
input = json.dumps([{'name': 'text', 'type': 'string'}])
output = json.dumps([{'name': 'text', 'type': 'string'}])
# Load model from spec
signature = ModelSignature.from_dict({'inputs': input, 'outputs': output})

# %%
from pathlib import Path

CONDA_ENV = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent, "conda.yaml")
print(CONDA_ENV)

# %%
MODEL_ARTIFACT_PATH = 'inference_pipeline_model'
with mlflow.start_run(run_name="chapter07_wrapped_inference_pipeline") as dl_model_tracking_run:
    finetuned_model_uri = 'runs:/1290f813d8e74a249c86eeab9f6ed24e/model'
    inference_pipeline_uri = f'runs:/{dl_model_tracking_run.info.run_id}/{MODEL_ARTIFACT_PATH}'
    mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH, 
                            conda_env=CONDA_ENV, 
                            python_model=InferencePipeline(finetuned_model_uri, inference_pipeline_uri), 
                            signature=signature)     


# %%
input = {"text":["what a disappointing movie","Great movie", "Great movie", "很好看的电影"]}
input_df = pd.DataFrame(input)
input_df

# %%
with mlflow.start_run(run_name="chapter07_wrap_inference_pipeline") as dl_model_prediction_run:
    loaded_model = mlflow.pyfunc.load_model(inference_pipeline_uri)
    results = loaded_model.predict(input_df)

# %%p
for i in range(results.size):
    print(results['text'][i])
# %%
