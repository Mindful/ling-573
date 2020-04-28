# ling-573 Multi-document Summarization System
UW CLMS program project for ling-573 by Paige Finkelstein, Jacob Hoffman, Wesley Rose, and Joshua Tanner

<br>

## Running the full system (summarization and ROUGE evaluation) on patas
All that is needed to run the end-to-end system on the `devtest` dataset on patas is to submit D2.cmd to condor:

`condor_submit D2.cmd`

This will run the **run_d2.sh** script, which will set up a conda environment and install all of the necessary dependencies. It will write the generated summaries to the **output** directory, and will write the ROUGE evalulation score report to **results/rouge_scores.out**.


##### To switch from running on the **dev_test** corpus to the **train** corpus
In **src/baseline.py** [here](https://github.com/Mindful/ling-573/blob/master/src/baseline.py#L11), uncomment `#topics = get_dataset_topics(TRAIN)` and comment out the line `topics = get_dataset_topics(DEV_TEST)` instead.

<br>

### Generating summarization outputs 
To run the summarization system and output summaries (to *outputs*):

`python3 src/baseline.py`

### Producing ROUGE score evaluations for existing outputs
Once you have summary results in the **outputs** directory, on patas run:

`sh eval.sh`

If not running on patas, be sure to switch the file paths in **eval.sh** to point at your correct local paths.
