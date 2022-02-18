# Instructions for Chapter 7

## Set up conda virtual environment dl-model-chapter07 by running:
   1. conda env create -f conda.yaml
   2. conda activate dl-model-chapter07

## Set up a local full-fledged MLflow tracking and registry server by following the instructions in Chapter 3.

## When running locally using the local full-fledged MLflow tracking server, make sure to set up the environmental variables as follows:
      export MLFLOW_TRACKING_URI=http://localhost
      export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
      export AWS_ACCESS_KEY_ID="minio"
      export AWS_SECRET_ACCESS_KEY="minio123"


