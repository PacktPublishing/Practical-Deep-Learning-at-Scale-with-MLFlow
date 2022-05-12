# %%
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
import pandas as pd


# %%
spark = SparkSession.builder.appName("Batch explanation with MLflow DL explainer").getOrCreate()

# %%
# use the run_id to construct a logged_model URI. An example is shown here:
run_id = "ad1edb09e5ea4d8ca0332b8bc2f5f6c9"
logged_explainer = f'runs:/{run_id}/nlp_sentiment_classifier_explainer'


# Load model as a Spark UDF.
explainer = mlflow.pyfunc.spark_udf(spark, model_uri=logged_explainer, result_type=StringType())


explainer

# %%
import datasets

dataset = datasets.load_dataset("imdb", split="test")

# shorten the strings to fit into the pipeline model
short_data = [v[:500] for v in dataset["text"][:20]]
# %%
df_pandas = pd.DataFrame (short_data, columns = ['text'])
spark_df = spark.createDataFrame(df_pandas)

# %%
spark_df = spark_df.withColumn('shap_values', explainer())

# %%
spark_df.show(n = 2, truncate=80, vertical=True)
# %%
