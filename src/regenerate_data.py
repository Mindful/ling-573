import os
import glob
import sys
from data import DATA_DIR, get_dataset_topics, configure_local, get_dataset_pickle_location
from common import Globals

'''
Cleans out the data/ directory and regenerates the results there. Must either be passed a local directory including 
all the corpora directories and topic config files, or run on patas with no arguments.
'''



def main():
    if len(sys.argv) > 1:
        local_dir = sys.argv[1]
        print('Running on local directory', local_dir, '...')
        configure_local(local_dir)



    print('Deleting existing data...')
    files = [get_dataset_pickle_location(dataset) for dataset in Globals.datasets]
    for f in files:
        if os.path.exists(f):
            os.remove(f)

    for dataset in Globals.datasets:
        print('Regenerating data for dataset', dataset)
        get_dataset_topics(dataset)

    print('Done')


if __name__ == '__main__':
    main()

