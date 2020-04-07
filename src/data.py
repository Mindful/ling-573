import lxml.etree as ET
import os.path
from abc import ABC
import re
from datetime import date


class Corpora(ABC):
    def __init__(self, name, start_year, end_year, directory):
        self.year_range = range(start_year, end_year+1)  # include final year
        self.name = name
        self.directory = directory

    def get_journal_dir(self, journal_id):
        raise RuntimeError("NYI")

    def get_filename(self, file_id, journal_id):
        raise RuntimeError("NYI")

    def get_file_location(self, file_id, journal_id):
        return os.path.join(self.get_journal_dir(journal_id), self.get_filename(file_id, journal_id))


class Aquaint(Corpora):
    def __init__(self):
        super().__init__('AQUAINT', 1996, 2000, '/corpora/LDC/LDC02T31/')

    def get_journal_dir(self, journal_id):
        return os.path.join(self.directory, journal_id.lower())

    def get_filename(self, file_id, journal_id):
        return ''.join([file_id, '_', journal_id])


class Aquaint2(Corpora):

    lang_id = 'eng'

    def __init__(self):
        super().__init__('AQUAINT-2', 2004, 2006, '/corpora/LDC/LDC08T25/data/')

    def get_journal_dir(self, journal_id):
        return os.path.join(self.directory, journal_id.lower() + '_' + self.lang_id)

    def get_filename(self, file_id, journal_id):
        return ''.join([journal_id.lower(), '_', self.lang_id, '_', file_id ])


CORPORA = [Aquaint(), Aquaint2()]


def get_child(parent, child_tag):
    return next(child for child in parent if child.tag == child_tag)


class TopicMetadata:
    ID = 'id'
    TITLE = 'title'
    NARRATIVE = 'narrative'
    DOCSET = 'docsetA'
    # docsetB not used for our task

    def __init__(self, topic_xml):
        self.id = topic_xml.attrib[self.ID]
        self.title = get_child(topic_xml, self.TITLE).text.strip()
        self.narrative= get_child(topic_xml, self.NARRATIVE).text.strip()
        self.docset = [x.attrib[self.ID] for x in get_child(topic_xml, self.DOCSET)]


class ArticleQuery:
    def __init__(self, text, topic_id):
        querystring, self.article_id = text.split('.')
        self.journal_id, self.lang, self.file_id = querystring.split('_')
        self.topic_id = topic_id
        self.year = int(self.file_id[0:4])

    def __repr__(self):
        return self.__dict__.__repr__()


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
        file = corpus.get_file_location(query.file_id, query.journal_id)
        if file not in queries_by_file:
            queries_by_file[file] = []

        queries_by_file[file].append(query)

    return queries_by_file


class Article:

    digit_regex = re.compile(r'\d')

    def __init__(self, id, type, headline, paragraphs):
        self.id = id

        date_ind = self.digit_regex.search(id).span()[0]
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
        id = get_child(xml_object, 'DOCNO').text.strip()
        type = get_child(xml_object, 'DOCTYPE').text.strip()
        headline = get_child(xml_object, 'HEADER').text.strip()
        paragraphs = [x.text.strip() for x in get_child(get_child(xml_object, 'BODY'), 'TEXT')]

        return Article(id, type, headline, paragraphs)


    @staticmethod
    def from_new_xml(xml_object):
        id = xml_object.attrib['id']
        type = xml_object.attrib['type']
        headline = get_child(xml_object, 'HEADLINE').text.strip()
        paragraphs = [x.text.strip() for x in get_child(xml_object, 'TEXT')]

        return Article(id, type, headline, paragraphs)


def read_new_content_file(filename):
    root = ET.parse(filename).getroot()

    return [Article.from_new_xml(x) for x in root]


def read_old_content_file(filename):
    parser = ET.XMLParser(recover=True)
    with open(filename, 'r') as f:
        file_content = f.read()

    root = ET.fromstring('<dummy>'+file_content+'</dummy>', parser=parser)

    return [Article.from_old_xml(x) for x in root]





