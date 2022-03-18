import mlflow.sagemaker

region='us-west-2'
role = "arn:aws:iam::565251169546:role/AWSSagemakerExecutionRole"
image_uri = '565251169546.dkr.ecr.us-west-2.amazonaws.com/mlflow-dl-inference:1.23.1'
# The MLflow model to deploy to SageMaker.
model_uri = 'dbfs:/databricks/mlflow-tracking/3340168740497031/cc8d2ad5b38d4d99bcb2b6b14af96b36/artifacts/inference_pipeline_model'

# s3 bucket for Sagemaker deployment. Sagemaker will copy the MLflow model from model_uri and put it in this bucket for deployment
bucket_for_sagemaker_deployment = 'dl-inference-deployment'
endpoint_name = 'dl-sentiment-model'

mlflow.sagemaker.deploy(
    mode='create',
    app_name=endpoint_name,
    model_uri=model_uri,
    image_url=image_uri,
    execution_role_arn=role,
    instance_type='ml.m5.xlarge',
    bucket = bucket_for_sagemaker_deployment,
    instance_count=1,
    region_name=region
)
