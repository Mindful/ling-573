import os
import pickle
from data.article import ArticleQuery, Article
from data.corpora import Aquaint, Aquaint2
from data.topic import Topic, read_topics_file
from progress.bar import Bar
from common import *
import re


DATA_DIR = os.path.join(ROOT_DIR, 'data/')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs/')

OUTPUT_FILE_STRING = '-A.M.100.'
OUTPUT_FILE_REGEX = re.compile(r'.*' + re.escape(OUTPUT_FILE_STRING) + r'.*')


DEV_TEST = 'dev_test'
TRAIN = 'train'


class DataManager:
    corpora = [Aquaint(), Aquaint2()]
    datasets = {
        DEV_TEST: '/dropbox/19-20/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml',
        TRAIN: '/dropbox/19-20/573/Data/Documents/training/2009/UpdateSumm09_test_topics.xml'
    }


def _compute_queries_by_file(topic_datas):
    queries = [ArticleQuery(article, topic.id) for topic in topic_datas for article in topic.docset]

    corpora_by_year = {}
    for corpus in DataManager.corpora:
        for year in corpus.year_range:
            corpora_by_year[year] = corpus

    queries_by_file = {}
    for query in queries:
        corpus = corpora_by_year[query.year]
        file = corpus.get_file_location(query)
        if file not in queries_by_file:
            queries_by_file[file] = []

        queries_by_file[file].append(query)

    return queries_by_file


def _fetch_articles_into_topics(queries, topic_metadatas):
    articles_by_topic = {}
    bar = Bar('Reading files...', max=len(queries))
    for corpus_file, query_list in queries.items():
        file_articles = corpus_file.get_articles()
        article_id_to_query = {q.article_id: q for q in query_list}
        for article in file_articles:
            article_id = article.id[-4:]
            if article_id in article_id_to_query:
                topic = article_id_to_query[article_id].topic_id
                if topic not in articles_by_topic:
                    articles_by_topic[topic] = []

                articles_by_topic[topic].append(article)
                del article_id_to_query[article_id]
        bar.next()

    topic_metadata_by_id = {t.id: t for t in topic_metadatas}

    topics = [
        Topic(topic_metadata_by_id[topic_id], article_list) for topic_id, article_list in articles_by_topic.items()
    ]

    bar.finish()

    return topics


def _write_out_summary(topic_id, summary_sentences, alphanum_id='1'):
    id_part_1 = topic_id[0:5]
    id_part_2 = topic_id[5:]
    output_filename = id_part_1 + OUTPUT_FILE_STRING + id_part_2 + '.' + alphanum_id

    with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
        f.writelines(x+'\n' for x in summary_sentences)


def output_summary(realization):
    _write_out_summary(realization.doc_group.topic_id, realization.summary)
    pass


def get_dataset_topics(dataset):
    if dataset not in DataManager.datasets:
        raise RuntimeError("Unknown dataset, please use one of the dataset constants.")

    dataset_location = DataManager.datasets[dataset]
    pickle_name = os.path.basename(dataset_location) + '.pickle'
    local_location = os.path.join(DATA_DIR, pickle_name)

    try:
        with open(local_location, 'rb') as picklefile:
            return pickle.load(picklefile)
    except FileNotFoundError:
        try:
            topic_metadatas = read_topics_file(dataset_location)
            queries = _compute_queries_by_file(topic_metadatas)
            topics = _fetch_articles_into_topics(queries, topic_metadatas)
            with open(local_location, 'wb') as picklefile:
                pickle.dump(topics, picklefile)
            return topics
        except OSError:
            raise RuntimeError("Failed to retrieve the base file. If you are running without the pickled file, please "
                               "make sure to run on patas")

