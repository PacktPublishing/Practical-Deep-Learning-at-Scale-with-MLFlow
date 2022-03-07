import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("Batch inference with MLflow DL inference pipeline").getOrCreate()

# load a logged model with run id and model name
# logged_model = 'runs:/37b5b4dd7bc04213a35db646520ec404/inference_pipeline_model'
# or load a registered model with a version number
#  model_uri=f"models:/{model_name}/{model_version}"

logged_model = 'models:/inference_pipeline_model/6'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type=StringType())

# pyfunc_udf = mlflow.pyfunc.spark_udf(<path-to-model-with-signature>)
# df = spark_df.withColumn("prediction", pyfunc_udf())

# Predict on a Spark DataFrame.
df = spark.read.csv('../data/imdb/test.csv', header=True)
df = df.select('review').withColumnRenamed('review', 'text')
df = df.withColumn('predictions', loaded_model())

df.show(n = 10, truncate=80, vertical=True)