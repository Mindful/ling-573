#!/bin/bash

conda init bash
source ~/.bashrc
source ~/.bash_profile
conda create --name summarization python=3.7 -y
conda activate summarization
pip install -r requirements.txt
python3 src/baseline.py

python3 src/generate_rouge_config.py  --peer_root_dir outputs/  --model_dir /dropbox/19-20/573/Data/models/devtest/
/dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl -e /dropbox/19-20/573/code/ROUGE/data -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d outputs/rouge_config.xml > results/rouge_scores.out
