# Instructions for Chapter 5

## Set up conda virtual environment dl_model by running:
   1. conda create -n dl_model python==3.8.10
   2. conda activate dl_model
   3. pip install -r requirements.txt
## Set up a local full-fledged MLflow tracking and registry server by following the instructions in Chapter 3. Make sure to set up the environment variables as follows before you run any command line tools:

      export MLFLOW_TRACKING_URI=http://localhost
      export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
      export AWS_ACCESS_KEY_ID="minio"
      export AWS_SECRET_ACCESS_KEY="minio123"
