import re
import spacy
from spacy.tokens import Doc, Span
from . import clean_text
from common import PipelineComponent, Globals
import re

ARTICLE_SENTENCE = 'article_sentence'
ARTICLE_HEADLINE = 'article_headline'
NARRATIVE = 'narrative'
TOPIC_TITLE = 'topic_title'

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

def contains_quote(span):
    scare_quotes = re.match(r".*[A-Za-z]+ \"\w+( \w+)?( \w+)?\" [A-Za-z]+.*", span.text)
    if scare_quotes:
        return False
    return "\"" in span.text


class DocumentGroup(PipelineComponent):
    __slots__ = ['topic_id', 'narrative', 'title', 'articles']

    @staticmethod
    def setup():
        Span.set_extension('contains_quote', getter=contains_quote)
        Span.set_extension('sent_index', default=-1)
        Span.set_extension('type', default=ARTICLE_SENTENCE)
        Doc.set_extension('paragraph_index', default=None)
        Globals.nlp.add_pipe(set_custom_boundaries, before='parser')

    def __init__(self, topic):
        self.topic_id = topic.id
        self.narrative = process_span(topic.narrative, NARRATIVE)
        self.title = process_span(topic.title, TOPIC_TITLE)
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
        self.headline = process_span(article.headline, ARTICLE_HEADLINE)
        self.type = article.type
        self.paragraphs = self._process_paragraphs(article.paragraphs)

    def __str__(self):
        return str({attr: getattr(self, attr) for attr in self.__slots__})

    def __repr__(self):
        return "<{} {}: {}>".format(self.__class__.__name__, self.id, self.date)

    def _process_paragraphs(self, paragraphs):
        cleaned = [clean_text(p) for p in paragraphs]
        return self._process_with_index_in_article(cleaned)

    def _process_with_index_in_article(self, paragraphs):
        paragraphs = list(enumerate(filter(None, paragraphs)))
        processed_paragraphs = [self._process_doc(p, i) for i, p in paragraphs]
        self._add_sent_indices(processed_paragraphs)
        return processed_paragraphs

    def _process_doc(self, paragraph, index):
        doc = Globals.nlp(paragraph)
        doc._.paragraph_index = index
        return doc

    def _add_sent_indices(self, paragraphs):
        index_counter = 0
        for p in paragraphs:
            for s in list(p.sents):
                s._.sent_index = index_counter
                index_counter += 1


def process_span(text, span_type):
    if text:
        span = Globals.nlp(text)[:]
        span._.type = span_type
        return span

    return None
