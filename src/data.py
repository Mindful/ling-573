import lxml.etree as ET
import os.path
from abc import ABC
import re
from datetime import date
from progress.bar import Bar
import pickle

DATA_DIR = '../data/'  # relative to source directory

digit_regex = re.compile(r'\d')


def read_new_content_file(filename):
    root = ET.parse(filename).getroot()

    return [Article.from_new_xml(x) for x in root]


def read_old_content_file(filename):
    parser = ET.XMLParser(recover=True)
    with open(filename, 'r') as f:
        file_content = f.read()

    root = ET.fromstring('<dummy>'+file_content+'</dummy>', parser=parser)

    return [Article.from_old_xml(x) for x in root]


class CorpusFile:
    def __init__(self, location, read_function):
        self.location = location
        self.read_function = read_function

    def get_articles(self):
        return self.read_function(self.location)

    def __eq__(self, other):
        return isinstance(other, CorpusFile) and self.location == other.location

    def __hash__(self):
        return hash(self.location)

    def __repr__(self):
        return "CorpusFile('{}')".format(self.location)


class Corpus(ABC):
    def __init__(self, name, start_year, end_year, directory):
        self.year_range = range(start_year, end_year+1)  # include final year
        self.name = name
        self.directory = directory

    def get_journal_dir(self, query):
        raise RuntimeError("NYI")

    def get_filename(self, query):
        raise RuntimeError("NYI")

    def get_file_location(self, query):
        file_location = os.path.join(self.get_journal_dir(query), self.get_filename(query))
        return CorpusFile(file_location, self.reader_function)


class Aquaint(Corpus):
    def __init__(self):
        super().__init__('AQUAINT', 1996, 2000, '/corpora/LDC/LDC02T31/')
        self.reader_function = read_old_content_file

    def get_journal_dir(self, query):
        return os.path.join(self.directory, query.journal_id.lower())

    def get_filename(self, query):
        parent_dir = str(query.year)
        string_parts = [query.file_id, '_',
                        'XIN' if query.journal_id == 'XIE' else query.journal_id]
        if query.journal_id != 'NYT':
            string_parts.append('_ENG')
        return os.path.join(parent_dir, ''.join(string_parts))


class Aquaint2(Corpus):

    lang_id = 'eng'

    def __init__(self):
        super().__init__('AQUAINT-2', 2004, 2006, '/corpora/LDC/LDC08T25/data/')
        self.reader_function = read_new_content_file

    def get_journal_dir(self, query):
        return os.path.join(self.directory, query.journal_id.lower() + '_' + self.lang_id)

    def get_filename(self, query):
        return ''.join([query.journal_id.lower(), '_', self.lang_id, '_', query.file_id[:-2], '.xml'])


CORPORA = [Aquaint(), Aquaint2()]


def get_child(parent, child_tag):
    try:
        return next(child for child in parent if child.tag == child_tag)
    except StopIteration:
        return None


def get_child_text(parent, child_tag):
    child = get_child(parent, child_tag)
    return child.text.strip() if child is not None else None


class TopicMetadata:
    ID = 'id'
    TITLE = 'title'
    NARRATIVE = 'narrative'
    DOCSET = 'docsetA'
    # docsetB not used for our task

    def __init__(self, topic_xml):
        self.id = topic_xml.attrib[self.ID]
        self.title = get_child_text(topic_xml, self.TITLE)
        self.narrative= get_child_text(topic_xml, self.NARRATIVE)
        self.docset = [x.attrib[self.ID] for x in get_child(topic_xml, self.DOCSET)]


class ArticleQuery:
    def __init__(self, text, topic_id):
        querystring, self.article_id = text.split('.')
        if querystring.find('_') == -1:
            # squished ID format, needs to be handled differently
            num_start = digit_regex.search(querystring).span()[0]
            self.journal_id, self.file_id = querystring[:num_start], querystring[num_start:]
        else:
            self.journal_id, _, self.file_id = querystring.split('_')

        self.topic_id = topic_id
        self.year = int(self.file_id[0:4])

    def __repr__(self):
        return self.__dict__.__repr__()


class Article:
    def __init__(self, id, type, headline, paragraphs):
        self.id = id

        date_ind = digit_regex.search(id).span()[0]
        year = int(id[date_ind:date_ind+4])
        month = int(id[date_ind+4:date_ind+6])
        day = int(id[date_ind+6:date_ind+8])

        self.date = date(year=year, month=month, day=day)
        self.type = type
        self.headline = headline
        self.paragraphs = paragraphs

    def __repr__(self):
        return self.__dict__.__repr__()

    @staticmethod
    def from_old_xml(xml_object):
        id = get_child_text(xml_object, 'DOCNO')
        type = get_child_text(xml_object, 'DOCTYPE')
        headline = get_child_text(xml_object, 'HEADER')
        paragraphs = [x.text.strip() for x in get_child(get_child(xml_object, 'BODY'), 'TEXT')]

        return Article(id, type, headline, paragraphs)


    @staticmethod
    def from_new_xml(xml_object):
        id = xml_object.attrib['id']
        type = xml_object.attrib['type']
        headline = get_child_text(xml_object, 'HEADLINE')
        paragraphs = [x.text.strip() for x in get_child(xml_object, 'TEXT')]

        return Article(id, type, headline, paragraphs)


class Topic:
    def __init__(self, metadata, articles):
        self.id = metadata.id
        self.title = metadata.title
        self.narrative = metadata.narrative
        self.articles = articles

    def __repr__(self):
        return '<{} {}: "{}">'.format(self.__class__.__name__, self.id, self.title)


def read_topics_file(filename):
    tree = ET.parse(filename)
    return [TopicMetadata(x) for x in tree.getroot()]


def compute_queries_by_file(topic_datas):
    queries = [ArticleQuery(article, topic.id) for topic in topic_datas for article in topic.docset]

    corpora_by_year = {}
    for corpus in CORPORA:
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


def fetch_articles_into_topics(queries, topic_metadatas):
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



def build_topics_from_file(filename):
    topic_metadatas = read_topics_file(filename)
    queries = compute_queries_by_file(topic_metadatas)
    topics = fetch_articles_into_topics(queries, topic_metadatas)
    return topics

DEV_TEST = 'dev_test'
TRAIN = 'train'

DATASETS = {
    DEV_TEST: '/dropbox/19-20/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml',
    TRAIN: '/dropbox/19-20/573/Data/Documents/training/2009/UpdateSumm09_test_topics.xml'
}


def get_dataset_topics(dataset):
    if dataset not in DATASETS:
        raise RuntimeError("Unknown dataset, please use one of the dataset constants.")

    dataset_location = DATASETS[dataset]
    pickle_name = os.path.basename(dataset_location) + '.pickle'
    local_location = os.path.join(DATA_DIR, pickle_name)

    try:
        with open(local_location, 'rb') as picklefile:
            return pickle.load(picklefile)
    except FileNotFoundError:
        try:
            topic_metadatas = read_topics_file(dataset_location)
            queries = compute_queries_by_file(topic_metadatas)
            topics = fetch_articles_into_topics(queries, topic_metadatas)
            with open(pickle_name, 'wb') as picklefile:
                pickle.dump(topics, picklefile)
            return topics
        except OSError:
            raise RuntimeError("Failed to retrieve the base file. If you are running without the pickled file, please"
                               "make sure to run on patas")



