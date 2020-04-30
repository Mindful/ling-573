import re
import spacy
from spacy.tokens import Doc, Span
from . import clean_text

def set_custom_boundaries(doc):
    '''
    Prevent sentence segmentation from splitting a quotation
    '''
    in_progress_quote = False
    for token in doc:
        if ("\"" in token.text) and (not in_progress_quote):
            in_progress_quote = True
        elif ("\"" in token.text) and in_progress_quote:
            in_progress_quote = False
            token.is_sent_start = False
        elif in_progress_quote:
            token.is_sent_start = False
    return doc


def english_nlp():
    print('Loading spaCy, this may take a moment.') #TODO: replace this with logging once logging is set up
    quote_getter = lambda span: "\"" in span.text
    Span.set_extension('contains_quote', getter=quote_getter)

    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe(set_custom_boundaries, before='parser')
    return nlp


nlp_parser = english_nlp()


class DocumentGroup:
    __slots__ = ['topic_id', 'narrative', 'title', 'articles']

    def __init__(self, topic):
        self.topic_id = topic.id
        self.narrative = process_text(topic.narrative)
        self.title = topic.title
        self.articles = [DocGroupArticle(article) for article in topic.articles]

    def __str__(self):
        return str({attr: getattr(self, attr) for attr in self.__slots__})

    def __repr__(self):
        return "<{} {}: {}>".format(self.__class__.__name__, self.topic_id, self.title)


class DocGroupArticle:
    __slots__ = ['id', 'date', 'headline', 'type', 'paragraphs']

    def __init__(self, article):
        self.id = article.id
        self.date = article.date
        self.headline = process_text(article.headline)
        self.type = article.type
        self.paragraphs = self._process_paragraphs(article.paragraphs)

    def __str__(self):
        return str({attr: getattr(self, attr) for attr in self.__slots__})

    def __repr__(self):
        return "<{} {}: {}>".format(self.__class__.__name__, self.id, self.date)

    def _process_paragraphs(self, paragraphs):
        cleaned = [clean_text(p) for p in paragraphs]
        return [nlp_parser(p) for p in cleaned if p]


def process_text(text):
    if text:
        return nlp_parser(text)
    return None
