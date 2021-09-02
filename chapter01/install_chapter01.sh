#!/bin/bash
conda create -n dl_model python==3.8.10
conda activate dl_model
pip install 'lightning-flash[all]'
