# Instructions for Chapter 10

## Set up a local full-fledged MLflow tracking and registry server by following the instructions in Chapter 3.

## When running locally using the local full-fledged MLflow tracking server, make sure to set up the environmental variables as follows.These are session-based, so each time when you start a new terminal window, make sure you run the following again before you run any code in this chapter.

      export MLFLOW_TRACKING_URI=http://localhost
      export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
      export AWS_ACCESS_KEY_ID="minio"
      export AWS_SECRET_ACCESS_KEY="minio123"
      export MLFLOW_EXPERIMENT_NAME=dl_explain_chapter10

## Set up conda virtual environment chapter10-dl-explain by running following. This virtual environment is for non PySpark UDF explainer:
       conda env create -f conda.yaml
       conda activate chapter10-dl-explain

## Batch explanation at-scale using PySpark UDF function

In addition to the local MLflow server setup and environment variables,
you also need to set up PySpark related virtual environment and 
environment variables as follows before running the PySpark UDF explainer.
### Set up conda virtual environment chapter10-dl-pyspark-explain by running:
       conda env create -f conda-pyspark-explain.yaml
       conda activate chapter10-dl-pyspark-explain
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

