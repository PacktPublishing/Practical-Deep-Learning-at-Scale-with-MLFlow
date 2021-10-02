# Code for Chapter 3

## Set up a local full-fledged MLflow tracking and registry server with MySQL as backend storage and MinIO as the artifact store
   1. Checkout the entire repo and switch to folder `chapter03/mlflow_docker_setup`
   2. Run `bash start_mlflow.sh` to start the mlflow server
      
      a) You must have docker running in your local environment in order to successfully run the docker-compose command
      
      b) Please refer to https://docs.docker.com/engine/install/ to install either Docker Desktop or a linux distr of docker. 
      
      c) Once the docker is installed, start it so it is running in the background.
   3. See the MLflow server UI at `http://localhost`
   4. See the object store (minIO, a multi-cloud object store) UI at `http://localhost:9000`
   5. When you are done, run `bash stop_mlflow.sh` to stop the local mlflow server
