#!/bin/bash
# must be run on patas in ling-573 repo root directory in order to work correctly
echo "Generating rouge config file..."
python3 src/generate_rouge_config.py  --peer_root_dir outputs/  --model_dir /dropbox/19-20/573/Data/models/devtest/
echo
echo "Rouge scores using most favorable comparison among models:"
echo
/dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl  -d -a -f B -e /dropbox/19-20/573/code/ROUGE/data outputs/rouge_config.xml
echo
echo "Rouge scores using average comparison among models:"
echo
/dropbox/19-20/573/code/ROUGE/ROUGE-1.5.5.pl  -d -a -f A -e /dropbox/19-20/573/code/ROUGE/data outputs/rouge_config.xml
