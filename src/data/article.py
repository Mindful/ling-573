import re
from datetime import date
from data.util import *

DIGIT_REGEX = re.compile(r'\d')


class ArticleQuery:
    def __init__(self, text, topic_id):
        self.full_article_id = text
        querystring, self.article_id = text.split('.')
        if querystring.find('_') == -1:
            # squished ID format, needs to be handled differently
            num_start = DIGIT_REGEX.search(querystring).span()[0]
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

        date_ind = DIGIT_REGEX.search(id).span()[0]
        year = int(id[date_ind:date_ind+4])
        month = int(id[date_ind+4:date_ind+6])
        day = int(id[date_ind+6:date_ind+8])

        self.date = date(year=year, month=month, day=day)
        self.type = type
        self.headline = headline
        self.paragraphs = paragraphs

    def get_raw_text(self):
        return ' '.join(self.paragraphs)

    def __repr__(self):
        return '<{} {}: "{}">'.format(self.__class__.__name__, self.id, self.headline)

    @staticmethod
    def from_old_xml(xml_object):
        id = get_child_text(xml_object, 'DOCNO')
        type = get_child_text(xml_object, 'DOCTYPE')

        body_obj = get_child(xml_object, 'BODY')
        headline = get_child_text(body_obj, 'HEADLINE')
        text_obj = get_child(body_obj, 'TEXT')
        paragraphs = [x.text.strip() for x in text_obj if x.text is not None]

        raw_text = text_obj.text.strip()
        if raw_text:
            paragraphs = [raw_text] + paragraphs

        return Article(id, type, headline, paragraphs)


    @staticmethod
    def from_new_xml(xml_object):
        id = xml_object.attrib['id']
        type = xml_object.attrib['type']
        headline = get_child_text(xml_object, 'HEADLINE')
        text_obj = get_child(xml_object, 'TEXT')
        paragraphs = [x.text.strip() for x in text_obj if x.text is not None]

        raw_text = text_obj.text.strip()
        if raw_text:
            paragraphs = [raw_text] + paragraphs


        return Article(id, type, headline, paragraphs)
