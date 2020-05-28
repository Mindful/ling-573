import pickle
from data.article import ArticleQuery, Article
from data.corpora import Aquaint, Aquaint2, Gigaword
from data.topic import Topic, read_topics_file
from progress.bar import Bar
from common import ROOT_DIR, Globals
import os
import re


DATA_DIR = os.path.join(ROOT_DIR, 'data/')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs/')

OUTPUT_FILE_STRING = '-A.M.100.'
OUTPUT_FILE_REGEX = re.compile(r'.*' + re.escape(OUTPUT_FILE_STRING) + r'.*')


def configure_local(directory):
    for name, location in Globals.datasets.items():
        Globals.datasets[name] = os.path.join(directory, os.path.basename(location))

    Globals.corpora = [Aquaint(directory), Aquaint2(directory), Gigaword(directory)]


def _compute_queries_by_file(topic_datas):
    queries = [ArticleQuery(article, topic.id) for topic in topic_datas for article in topic.docset]

    corpora_by_year = {}
    for corpus in Globals.corpora:
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
        article_id_to_query = {q.full_article_id: q for q in query_list}
        for article in file_articles:
            full_article_id = article.id
            if full_article_id in article_id_to_query:
                topic = article_id_to_query[full_article_id].topic_id
                if topic not in articles_by_topic:
                    articles_by_topic[topic] = []

                articles_by_topic[topic].append(article)
                del article_id_to_query[full_article_id]
        assert(len(article_id_to_query) == 0)  # We must get all the articles we're looking for
        bar.next()

    topic_metadata_by_id = {t.id: t for t in topic_metadatas}

    topics = [
        Topic(topic_metadata_by_id[topic_id], article_list) for topic_id, article_list in articles_by_topic.items()
    ]

    bar.finish()

    return topics


def _write_out_summary(topic_id, final_content, alphanum_id='1'):
    id_part_1 = topic_id[0:5]
    id_part_2 = topic_id[5:]
    output_filename = id_part_1 + OUTPUT_FILE_STRING + id_part_2 + '.' + alphanum_id

    with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
        f.writelines(x.realized_text +'\n' for x in final_content)


def output_summary(ordering):
    _write_out_summary(ordering.doc_group.topic_id, ordering.ordered_sents)


def load_all_articles(corpus):
    files = corpus.get_all_files()
    bar = Bar('Loading articles from files...', max=len(files))
    output = []
    for f in files:
        output.extend(f.get_articles())
        bar.next()

    bar.finish()
    return output


def load_sample_articles(corpus, count=20):
    files = corpus.get_all_files()[0:count]
    output = []
    for f in files:
        output.extend(f.get_articles())

    return output


def get_dataset_pickle_location(dataset):
    dataset_location = Globals.datasets[dataset]
    pickle_name = os.path.basename(dataset_location) + '.pickle'
    local_location = os.path.join(DATA_DIR, pickle_name)
    return local_location


def get_dataset_topics(dataset):
    if dataset not in Globals.datasets:
        raise RuntimeError("Unknown dataset, please use one of the dataset constants.")

    local_location = get_dataset_pickle_location(dataset)

    try:
        with open(local_location, 'rb') as picklefile:
            return pickle.load(picklefile)
    except FileNotFoundError:
        try:
            topic_metadatas = read_topics_file(Globals.datasets[dataset])
            queries = _compute_queries_by_file(topic_metadatas)
            topics = _fetch_articles_into_topics(queries, topic_metadatas)
            with open(local_location, 'wb') as picklefile:
                pickle.dump(topics, picklefile)
            return topics
        except OSError:
            raise RuntimeError("Failed to retrieve the base file. If you are running without the pickled file, please "
                               "make sure to run on patas")

