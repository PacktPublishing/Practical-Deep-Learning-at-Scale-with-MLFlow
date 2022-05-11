# Instructions for Chapter 10

## Set up a local full-fledged MLflow tracking and registry server by following the instructions in Chapter 3.

## When running locally using the local full-fledged MLflow tracking server, make sure to set up the environmental variables as follows:
      export MLFLOW_TRACKING_URI=http://localhost
      export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
      export AWS_ACCESS_KEY_ID="minio"
      export AWS_SECRET_ACCESS_KEY="minio123"

## Set up conda virtual environment chapter10-dl-explain by running:
       conda env create -f conda.yaml
       conda activate chapter10-dl-explain
