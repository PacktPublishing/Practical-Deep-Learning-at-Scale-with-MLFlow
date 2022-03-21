FROM mlflow-dl-inference
ADD mlruns/1/meta.yaml  /opt/mlflow/mlruns/1/meta.yaml
ADD mlruns/1/d01fc81e11e842f5b9556ae04136c0d3/ /opt/mlflow/mlruns/1/d01fc81e11e842f5b9556ae04136c0d3/
ADD tmp/opt/mlflow/hf/cache/dl_model_chapter08/csv/ /opt/mlflow/tmp/opt/mlflow/hf/cache/dl_model_chapter08/csv/
ARG hf_datasets_cache_path=tmp/opt/mlflow/hf/cache/dl_model_chapter08 
ENV HF_DATASETS_CACHE=$hf_datasets_cache_path
