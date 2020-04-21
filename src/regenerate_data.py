import os
import glob
import sys
from data import DATA_DIR, get_dataset_topics, DataManager
from data.corpora import Aquaint, Aquaint2

'''
Cleans out the data/ directory and regenerates the results there. Must either be passed a local directory including 
all the corpora directories and topic config files, or run on patas with no arguments.
'''


def configure_local(directory):
    for name, location in DataManager.datasets.items():
        DataManager.datasets[name] = os.path.join(directory, os.path.basename(location))

    DataManager.corpora = [Aquaint(directory), Aquaint2(directory)]


def main():
    if len(sys.argv) > 1:
        local_dir = sys.argv[1]
        print('Running on local directory', local_dir, '...')
        configure_local(local_dir)

    print('Deleting existing data...')
    files = glob.glob(DATA_DIR+'/*')
    for f in files:
        os.remove(f)

    for dataset in DataManager.datasets:
        print('Regenerating data for dataset', dataset)
        get_dataset_topics(dataset)

    print('Done')


if __name__ == '__main__':
    main()

