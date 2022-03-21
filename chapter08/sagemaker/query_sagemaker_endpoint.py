import boto3

def check_status(app_name):
  sage_client = boto3.client('sagemaker', region_name="us-west-2")
  endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
  return endpoint_description['EndpointStatus']

app_name = 'dl-sentiment-model'

print("Application status is: {}".format(check_status(app_name)))

if check_status(app_name) == 'InService':
    client = boto3.client('sagemaker-runtime')

    content_type = "application/json; format=pandas-split"                                        
    accept = "text/plain"                       
    payload = '{"columns": ["text"],"data": [["This is the best movie we saw."], ["What a movie!"]]}'

    response = client.invoke_endpoint(
        EndpointName=app_name, 
        ContentType=content_type,
        Accept=accept,
        Body=payload
        )

    print(response['Body'].read().decode('utf-8'))    