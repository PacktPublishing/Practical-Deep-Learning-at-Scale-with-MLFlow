# Practical Deep Learning at Scale with MLflow

<a href="https://www.packtpub.com/product/practical-deep-learning-at-scale-with-mlflow/9781803241333?utm_source=github&utm_medium=repository&utm_campaign=9781803241333"><img src="https://static.packt-cdn.com/products/9781803241333/cover/smaller" alt="Practical Deep Learning at Scale with MLflow" height="256px" align="right"></a>

This is the code repository for [Practical Deep Learning at Scale with MLflow](https://www.packtpub.com/product/practical-deep-learning-at-scale-with-mlflow/9781803241333?utm_source=github&utm_medium=repository&utm_campaign=9781803241333), published by Packt.

**Bridge the gap between offline experimentation and online production**

## What is this book about?
The book starts with an overview of the deep learning (DL) life cycle and the emerging Machine Learning Ops (MLOps) field, providing a clear picture of the four pillars of deep learning: 
data, model, code, and explainability and the role of MLflow in these areas.

This book covers the following exciting features: 
* Understand MLOps and deep learning life cycle development
* Track deep learning models, code, data, parameters, and metrics
* Build, deploy, and run deep learning model pipelines anywhere
* Run hyperparameter optimization at scale to tune deep learning models
* Build production-grade multi-step deep learning inference pipelines
* Implement scalable deep learning explainability as a service
* Deploy deep learning batch and streaming inference services
* Ship practical NLP solutions from experimentation to production

If you feel this book is for you, get your copy at [Amazon](https://www.amazon.com/Practical-Deep-Learning-Scale-MLflow/dp/1803241330/) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
Xclient = boto3.client('sagemaker-runtime')
response = client.invoke_endpoint(
EndpointName=app_name,
ContentType=content_type,
Accept=accept,
Body=payload
)
```

**Following is what you need for this book:**
This book is for machine learning practitioners including data scientists, data engineers, ML engineers, and scientists who want to build scalable full life cycle deep learning pipelines with reproducibility and provenance tracking using MLflow. 
A basic understanding of data science and machine learning is necessary to grasp the concepts presented in this book.

With the following software and hardware list you can run all code files present in the book (Chapter 1-10).

### Software and Hardware List


The majority of the code in this book can be implemented and executed using the open
source MLflow tool, with a few exceptions where a 14-day full Databricks trial is needed
(sign up at https://databricks.com/try-databricks) along with an AWS
Free Tier account (sign up at https://aws.amazon.com/free/). The following lists
some major software packages covered in this book:

* MLflow 1.20.2 and above
* Python 3.8.10
* Lightning-flash 0.5.0
* Transformers 4.9.2
* SHAP 0.40.0
* PySpark 3.2.1
* Ray[tune] 1.9.2
* Optuna 2.10.0

The complete package dependencies are listed in each chapter's requirements.txt
file or the conda.yaml file in this book's GitHub repository. All code has been tested
to run successfully in a macOS or Linux environment. If you are a Microsoft Windows
user, it is recommended to install WSL2 to run the bash scripts provided in this book:
https://www.windowscentral.com/how-install-wsl2-windows-10. 
It is a known issue that the MLflow CLI does not work properly in the Microsoft Windows
command line. 

Starting from Chapter 3, Tracking Models, Parameters, and Metrics of this book, you
will also need to have Docker Desktop (https://www.docker.com/products/
docker-desktop/) installed to set up a fully-fledged local MLflow tracking server for executing the code in this book. AWS SageMaker is needed in Chapter 8, Deploying a
DL Inference Pipeline at Scale, for the cloud deployment example. VS Code version 1.60 or above (https://code.visualstudio.com/updates/v1_60) is used as the
integrated development environment (IDE) in this book. Miniconda version 4.10.3 or above (https://docs.conda.io/en/latest/miniconda.html) is used
throughout this book for creating and activating virtual environments.

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781803241333_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Engineering MLOps [[Packt]](https://www.packtpub.com/product/engineering-mlops/9781800562882?utm_source=github&utm_medium=repository&utm_campaign=9781800562882) [[Amazon]](https://www.amazon.com/dp/B08PFN73CM)

* Machine Learning Engineering with Python [[Packt]](https://www.packtpub.com/product/machine-learning-engineering-with-python/9781801079259?utm_source=github&utm_medium=repository&utm_campaign=9781801079259) [[Amazon]](https://www.amazon.com/dp/B09CHHK2RJ)

## Get to Know the Author
**Yong Liu**
has been working in big data science, machine learning, and optimization since
his doctoral student years at the University of Illinois at Urbana-Champaign (UIUC)
and later as a senior research scientist and principal investigator at the National Center
for Supercomputing Applications (NCSA), where he led data science R&D projects
funded by the National Science Foundation and Microsoft Research. He then joined
Microsoft and AI/ML start-ups in the industry. He has shipped ML and DL models to
production and has been a speaker at the Spark/Data+AI summit and NLP summit.
He has recently published peer-reviewed papers on deep learning, linked data, and
knowledge-infused learning at various ACM/IEEE conferences and journals.
### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781803241333">https://packt.link/free-ebook/9781803241333 </a> </p>