# Instructions for Chapter 5

## Set up conda virtual environment dl_model by running:
   1. conda create -n dl_model python==3.8.10
   2. conda activate dl_model
   3. pip install -r requirements.txt
## Set up a local full-fledged MLflow tracking and registry server by following the instructions in Chapter 3. Make sure to set up the environment variables as follows before you run any command line tools:

export MLFLOW_TRACKING_URI=http://localhost:8080
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"


mlflow run . --experiment-name='/Shared/dl_model_chapter05' -P pipeline_steps='download_data'



export MLFLOW_TRACKING_URI=databricks
export DATABRICKS_HOST=https://adb-7148578107722284.4.azuredatabricks.net/
export DATABRICKS_TOKEN=dapi8729ec2e9bb053c9e14d4995275c0b11

mlflow run https://github.com/bibuwei/Practical-Deep-Learning-at-Scale-with-MLFlow#chapter05 -v a71c7d7f0c13ad646f25fc6691e58bd1153c360c  --experiment-name='/Shared/dl_model_chapter05' -P pipeline_steps='download_data'


https://github.com/bibuwei/Practical-Deep-Learning-at-Scale-with-MLFlow/tree/a71c7d7f0c13ad646f25fc6691e58bd1153c360c/chapter05
