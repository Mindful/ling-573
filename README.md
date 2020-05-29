# ling-573 Multi-document Summarization System
UW CLMS program project for ling-573 by Paige Finkelstein, Jacob Hoffman, Wesley Rose, and Joshua Tanner

<br>

## Running the full system on patas for Deliverable 4
All that is needed to run the end-to-end system on the `devtest` and `evaltest` datasets on patas is to submit D4.cmd to condor:

`condor_submit D4.cmd`

This will run the **run.sh** script, which will set up a conda environment and install all of the necessary dependencies. 

It will write the generated summaries for `devtest` to the **output/devtest/** directory, and write the generated summaries for `evaltest` to the **output/evaltest/** directory. 

The ROUGE evalulation score reports for both will be saved in the **results**  directory to as **rouge_scores_devtest.out** and **rouge_scores_evaltest.out**.

<br>

### Generating summarization outputs 
To run the summarization system and output summaries (to *outputs*):

`python3 src/baseline.py devtest|train|evaltest`

### Producing ROUGE score evaluations for existing outputs
Once you have summary results in the **outputs/x** directory, adjust the dataset name in the **eval.sh** file, and run:

`sh eval.sh`

If not running on patas, be sure to switch the file paths in **eval.sh** to point at your correct local paths.
