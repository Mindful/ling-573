from data import *
from preprocessing.topic_doc_group import DocumentGroup
# have to at least import Topic class from data for unpickling to work

def main():
    #train_topics = get_dataset_topics(TRAIN)
    dev_topics = get_dataset_topics(DEV_TEST)
    for topic in dev_topics:
        doc_group = DocumentGroup(topic)



if __name__ == '__main__':
    main()