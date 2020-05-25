from data import *  # needs to be import * for pickling to work
from preprocessing.topic_doc_group import DocumentGroup
from content_selection.selection import Selection
from information_ordering.ordering import Ordering
from content_realization.realization import Realization
from common import *
from os.path import join
from shutil import copyfile

def main():
    # topics = get_dataset_topics(TRAIN)
    topics = get_dataset_topics(DEV_TEST)

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
    main()
