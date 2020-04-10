import os
import glob
from data import DATA_DIR, get_dataset_topics, DATASETS

'''
Cleans out the data/ directory and regenerates the results there. Requires no arguments, but must be run on patas.
'''


def main():
    print('Deleting existing data...')
    files = glob.glob(DATA_DIR+'/*')
    for f in files:
        os.remove(f)

    for dataset in DATASETS:
        print('Regenerating data for dataset', dataset)
        get_dataset_topics(dataset)

    print('Done')


if __name__ == '__main__':
    main()

