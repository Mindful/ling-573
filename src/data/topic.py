import lxml.etree as ET
from data.util import *


def read_topics_file(filename):
    tree = ET.parse(filename)
    return [TopicMetadata(x) for x in tree.getroot()]


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


class Topic:
    def __init__(self, metadata, articles):
        self.id = metadata.id
        self.title = metadata.title
        self.narrative = metadata.narrative
        self.articles = articles

    def __repr__(self):
        return '<{} {}: "{}">'.format(self.__class__.__name__, self.id, self.title)