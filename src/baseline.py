from data import *  # needs to be import * for pickling to work
from preprocessing.topic_doc_group import DocumentGroup
from content_selection.selection import Selection
from progress.bar import Bar


def main():
    #topics = get_dataset_topics(TRAIN)
    topics = get_dataset_topics(DEV_TEST)

    bar = Bar('Processing topics...', max=len(topics))
    for topic in topics:
        doc_group = DocumentGroup(topic)
        #TODO: process the topic, then pass the final result into output_summary
        # Content selection
        #selected_content = Selection(doc_group)


        output_summary(None)
        bar.next()




if __name__ == '__main__':
    main()