from data import *  # needs to be import * for pickling to work
from preprocessing.topic_doc_group import DocumentGroup
from content_selection.selection import Selection
from information_ordering.ordering import Ordering
from content_realization.realization import Realization
from progress.bar import Bar


def main():
    #topics = get_dataset_topics(TRAIN)
    topics = get_dataset_topics(DEV_TEST)

    Selection.selection_method = Selection.select_ngram
    #Selection.selection_method = Selection.select_lexrank
    #Selection.selection_method = Selection.select_simple

    bar = Bar('Processing topics...', max=len(topics))
    for topic in topics:
        doc_group = DocumentGroup(topic)
        selected_content = Selection(doc_group)
        realized_content = Realization(selected_content)
        ordered_content = Ordering(realized_content, use_BERT=False)
        output_summary(ordered_content)
        bar.next()

    print()




if __name__ == '__main__':
    main()
