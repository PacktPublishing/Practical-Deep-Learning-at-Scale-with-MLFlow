import mlflow.sagemaker
import shutil

# make a copy of the mlruns to local /opt/mlflow folder to allow sagemaker deployment pick the model up
src = "mlruns/1"
dest = "/opt/mlflow/mlruns/1"
shutil.copytree(src, dest, dirs_exist_ok=True)

region='us-west-2'
# You need to create this role and assign proper permission. replace XXXXX with your own AWS account number
role = "arn:aws:iam::XXXXX:role/AWSSagemakerExecutionRole"
# docker image URI registered in AWS ECR. replace XXXXX with your own AWS account number
image_uri = 'XXXXX.dkr.ecr.us-west-2.amazonaws.com/mlflow-dl-inference-w-finetuned-model:1.23.1'
# The MLflow inference pipeline model to deploy to SageMaker.
model_uri = 'runs:/dc5f670efa1a4eac95683633ffcfdd79/inference_pipeline_model'

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
