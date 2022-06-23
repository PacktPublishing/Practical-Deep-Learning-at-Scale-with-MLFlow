# Instructions for Chapter 2
   0. install mlflow by running 'pip install mlflow' in your dl_model virtual environment from Chapter01.
      If you just get started, then run the following to set up virtual environment and install:
      
      conda create -n dl_model python==3.8.10
      
      conda activate dl_model
      
      pip install -r requirements.txt
   
   1. To run the first deep learning model with mlflow, type `python first_dl_with_mlflow.py` in your command line
   2. Start the mlflow UI web server at the command line `mlflow ui`
   3. Go to a local web browser and open the URL: `https://127.0.0.1:5000/` and you will see the mlflow UI with the logged DL model experiment run.
