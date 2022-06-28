# Code and Instructions for Chapter1

1. Installation of all dependencies

   a) Set up a new virtual environment dl_model and install lightnining-flash[all]==0.5.0. Note that due to the rapid change
      of lightning-flash API, sometimes backward incompatibility happens. We thus use this README to make all necessary updates.
      So instead of just installing lightning-flash, we use a standard python package requirements.txt file to freeze all related packages and versions.
      Run the following commands to get the dependencies installed on the virtual environment 'dl_model':
     
     
      conda create -n dl_model python==3.8.10
      
      conda activate dl_model
      
      pip install -r requirements.txt

      if you have a `NVIDIA GeForce RTX 3050 Ti Laptop GPU in a windows surface laptop with windows 11`, you need to install additional dependencies using the additional requirements file in this chapter's folder as follows:

      pip install -r requirements-gpu-additional.txt


   b) You need to have conda installed first before you run the above command

   c) Please refer to `https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation` for miniconda installation on your local environment
   
2. A first deep learning model `first_dl.py`. 
3. To run the first deep learning model, type `python first_dl.py` in your command line.
