# Instructions for Chapter 8

## Set up a local full-fledged MLflow tracking and registry server by following the instructions in Chapter 3.

## When running locally using the local full-fledged MLflow tracking server, make sure to set up the environmental variables as follows:
      export MLFLOW_TRACKING_URI=http://localhost
      export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
      export AWS_ACCESS_KEY_ID="minio"
      export AWS_SECRET_ACCESS_KEY="minio123"

## Batch inference at-scale using PySpark UDF function
### Set up conda virtual environment chapter08-batch-inference by running:
       conda env create -f conda-batch-inference.yaml
       conda activate chapter08-batch-inference
### Install Spark 3.2.1 on your local environment
   1. Download the spark-3.2.1-bin-hadoop3.2.tgz:
      https://www.apache.org/dyn/closer.lua/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
   2. Unpack the bundle in a folder called spark_server in your local environment, for example: /Users/yongliu/spark_server
      ```
      tar -xvf spark-3.2.1-bin-hadoop3.2.tgz
      ```
   3. Set up the following environment variables before you run any spark related batch inference job:
      ```
      export SPARK_HOME=/Users/yongliu/spark_server/spark-3.2.1-bin-hadoop3.2
      export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.3-src.zip:$PYTHONPATH
      export PATH=$SPARK_HOME/bin:$SPARK_HOME/python:$PATH
      ```
   4. Verify your local pypsark working by typing 'pyspark' on the command line. If the install works, you should see the following screen:
   ```> pyspark
Python 3.8.10 | packaged by conda-forge | (default, May 10 2021, 22:58:09)
[Clang 11.1.0 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
22/03/06 21:06:40 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 3.2.1
      /_/

Using Python version 3.8.10 (default, May 10 2021 22:58:09)
Spark context Web UI available at http://yongs-mbp.lan:4040
Spark context available as 'sc' (master = local[*], app id = local-1646629601965).
SparkSession available as 'spark'.
>>>
```
## Deploying to AWS Sagemaker: preparation
To reproduce what we have in the github repo, follow the steps below under the chapter08 folder:
   1.	Set up the following two environment variables:

      export MLFLOW_TRACKING_URI=
      export HF_DATASETS_CACHE=tmp/opt/mlflow/hf/cache/dl_model_chapter08

   The first environment variable turns off the fully-fledged local MLflow tracking server and uses the local file system as the tracking backend. The second environment variable specifies the Huggingface datasets cache location, which is used by the fine-tuned model. 

   2.	Run the fine-tuning step pipeline as follows:
      
      mlflow run . -e fine_tuning_model --experiment-name='dl_model_chapter08' -P data_path='./data'

   Once the fine tuning model step is done, use the run_id  (here we use an example run_id `d01fc81e11e842f5b9556ae04136c0d3`) as the parameter to run the inference_pipeline_model step as follows to produce the inference_pipeline_model:
      
      mlflow run . -e inference_pipeline_model  --experiment-name='dl_model_chapter08' -P finetuned_model_run_id='d01fc81e11e842f5b9556ae04136c0d3'
      
   This will produce two mlruns under chapter08 folder. Make sure you modify the `meta.yaml` to have the correct path needed for the docker image we want to build to use. You can refer to the `meta.yaml` file in the GitHub Repo. 
   As a reminder, you can also directly use the mlruns in the GitHub repo to do the deployment. 




