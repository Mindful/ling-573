#!/bin/bash

conda init bash
source ~/.bashrc
source ~/.bash_profile
conda create --name metrics python=3.7 -y
conda activate metrics
pip install -r requirements.txt

python3 src/metric_computation.py