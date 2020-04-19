from data import *  # needs to be import * for pickling to work
from preprocessing.topic_doc_group import DocumentGroup
from content_selection.selection import Selection
from information_ordering.ordering import Ordering
from content_realization.realization import Realization
from progress.bar import Bar


def main():
    #topics = get_dataset_topics(TRAIN)
    topics = get_dataset_topics(DEV_TEST)

    bar = Bar('Processing topics...', max=len(topics))
    for topic in topics:
        doc_group = DocumentGroup(topic)
        selected_content = Selection(doc_group)
        ordered_content = Ordering(selected_content)
        realized_content = Realization(ordered_content)
        output_summary(realized_content)
        bar.next()




if __name__ == '__main__':
    main()
