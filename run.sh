#!/bin/bash

conda init bash
source ~/.bashrc
source ~/.bash_profile
conda create --name summarization python=3.7 -y
conda activate summarization
pip install -r requirements.txt

python3 src/baseline.py devtest
python3 src/generate_rouge_config.py devtest --peer_root_dir outputs/devtest/  --model_dir /dropbox/19-20/573/Data/models/devtest/
/dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl -e /dropbox/19-20/573/code/ROUGE/data -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d outputs/devtest/rouge_config.xml > results/rouge_scores_devtest.out

python3 src/baseline.py evaltest
python3 src/generate_rouge_config.py evaltest --peer_root_dir outputs/evaltest/  --model_dir /dropbox/19-20/573/Data/models/evaltest/
/dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl -e /dropbox/19-20/573/code/ROUGE/data -a -n 2 -x -m -c 95 -r 1000 -f A -p 0.5 -t 0 -l 100 -s -d outputs/evaltest/rouge_config.xml > results/rouge_scores_evaltest.out


