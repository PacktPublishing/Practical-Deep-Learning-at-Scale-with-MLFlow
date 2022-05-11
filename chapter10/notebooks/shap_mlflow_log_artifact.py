# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# This is a VSCode Notebook. You can run it by opening it in VSCode and run interactively
#  or you can run it directly in the command line with 'python shap_mlflow_log_artifact.py'
# %%
import os
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import TempDir

import shap
import transformers
from shap.plots import *
import numpy as np

# %%
# local full-fledged MLflow server
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

# disable Hugging Face warning
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# %%
EXPERIMENT_NAME = "dl_explain_chapter10"
mlflow.set_tracking_uri('http://localhost')
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

# %%
# create a DL model that can classify a sentence into POSITIVE or NEGATIVE sentiment
dl_model = transformers.pipeline('sentiment-analysis', return_all_scores=False)
explainer = shap.Explainer(dl_model) 
shap_values = explainer(["Not a good movie to spend time on.", "This is a great movie."])

# %%
mlflow.end_run()
artifact_root_path = "model_explanations_shap"
artifact_file_name = 'shap_bar_plot'
with mlflow.start_run() as run:
    with TempDir() as temp_dir:
        temp_dir_path = temp_dir.path()
        print("temp directory for artifacts: {}".format(temp_dir_path))
        try:
            plt.clf()
            plt.subplots_adjust(bottom=0.2, left=0.4)
            shap.plots.bar(shap_values[0, :, "NEGATIVE"], show=False)
            plt.savefig(f"{temp_dir_path}/{artifact_file_name}")
        finally:
            plt.close(plt.gcf())
        np.save(f"{temp_dir_path}/shap_values", shap_values.values)
        np.save(f"{temp_dir_path}/base_values", shap_values.base_values)
        mlflow.log_artifact(f"{temp_dir_path}/shap_values.npy", artifact_root_path)
        mlflow.log_artifact(f"{temp_dir_path}/base_values.npy", artifact_root_path)
        mlflow.log_artifact(f"{temp_dir_path}/{artifact_file_name}.png", artifact_root_path)

# %%
# verify the shap bar plot is stored in the MLflow artifact store
print("run_id: {}".format(run.info.run_id))
artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id,
            artifact_root_path)]
print("artifacts: {}".format(artifacts))
# %%
# dowload artifacts from MLflow server to local directory
downloaded_local_path = MlflowClient().download_artifacts(run.info.run_id, artifact_root_path)
print("\n# downloaded_local_path:")

print(downloaded_local_path)
base_values = np.load(os.path.join(downloaded_local_path, "base_values.npy"), allow_pickle=True)
shap_values = np.load(os.path.join(downloaded_local_path, "shap_values.npy"), allow_pickle=True)

print("\n# base_values:")
print(base_values)
print("\n# shap_values:")
print(shap_values[:3])
# %%
