from mlflow.deployments import get_deploy_client
import pandas as pd
import json

target_uri = 'ray-serve'
plugin = get_deploy_client(target_uri)
print(plugin.list_deployments())

# set up a dataframe
df = pd.DataFrame([['What a Movie'], ['Great Movie']], columns=['text'])
response = plugin.predict(deployment_name='dl-inference-model-on-ray', df = df)
obj = json.loads(response.to_json(orient='split'))
  
# Pretty Print JSON
json_formatted_str = json.dumps(obj, indent=4)
print(json_formatted_str)