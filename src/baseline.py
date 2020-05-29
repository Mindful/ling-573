from data import *  # needs to be import * for pickling to work
from preprocessing.topic_doc_group import DocumentGroup
from content_selection.selection import Selection
from information_ordering.ordering import Ordering
from content_realization.realization import Realization
from common import *
from os.path import join
from shutil import copyfile
import sys

DATASETS = {
    'train': get_dataset_topics(TRAIN),
    'devtest': get_dataset_topics(DEV_TEST),
    'evaltest': get_dataset_topics(EVAL)
}

def main(dataset_name):
    topics = DATASETS[dataset_name]

    pipeline_classes = [
        DocumentGroup,
        Selection,
        Realization,
        Ordering,
    ]

    setup(pipeline_classes)

    for index, topic in enumerate(topics):
        Globals.logger.info('Processing document group {}/{}'.format(index+1, len(topics)))
        doc_group = DocumentGroup(topic)
        selected_content = Selection(doc_group)
        realized_content = Realization(selected_content)
        ordered_content = Ordering(realized_content)
        output_summary(ordered_content)

    Globals.logger.info("Writing config file for this run to output directory")
    copyfile(CONFIG_FILE, join(OUTPUT_DIR, CONFIG_FILENAME))



if __name__ == '__main__':
    dataset_name = sys.argv[1]

    if dataset_name not in DATASETS.keys():
        raise Exception("Invalid dataset given", "dataset options: {}".format(', '.join(DATASETS.keys())))

    main(dataset_name)
