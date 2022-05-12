# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pickle

import mlflow
import pandas as pd
import shap
from shap.plots import *

# %%
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

# %%
# use the run_id to construct a logged_model URI. An example is shown here:
run_id = "ad1edb09e5ea4d8ca0332b8bc2f5f6c9"
logged_explainer = f'runs:/{run_id}/nlp_sentiment_classifier_explainer'
explainer = mlflow.pyfunc.load_model(logged_explainer)

explainer

# %%
import datasets

dataset = datasets.load_dataset("imdb", split="test")

# shorten the strings to fit into the pipeline model
short_data = [v[:500] for v in dataset["text"][:20]]

# %%
# create a pandas dataframe to feed into MLflow logged explainer
df_test = pd.DataFrame (short_data, columns = ['text'])

# call the explainer predict function, which is to explain using SHAP
results = explainer.predict(df_test)
# %%
results_deserialized = pickle.loads(results['shap_values'][0])
print(results_deserialized)
# %%

shap.plots.text(results_deserialized[:,:,"POSITIVE"])

