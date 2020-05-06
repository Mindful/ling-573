import re
import spacy
from spacy.tokens import Doc, Span
from . import clean_text
from common import NLP

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

Span.set_extension('contains_quote', getter=lambda span: "\"" in span.text)
Span.set_extension('sent_index', default=None)
Doc.set_extension('paragraph_index', default=None)
NLP.add_pipe(set_custom_boundaries, before='parser')


class DocumentGroup:
    __slots__ = ['topic_id', 'narrative', 'title', 'articles']

    def __init__(self, topic):
        self.topic_id = topic.id
        self.narrative = process_span(topic.narrative)
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
        self.headline = self._process_headline(article.headline)
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
        doc = NLP(paragraph)
        doc._.paragraph_index = index
        return doc

    def _add_sent_indices(self, paragraphs):
        index_counter = 0
        for p in paragraphs:
            for s in list(p.sents):
                s._.sent_index = index_counter
                index_counter += 1

    def _process_headline(self, text):
        headline = process_span(text)
        if headline:
            headline._.sent_index = -1
        return headline



def process_span(text):
    if text:
        return NLP(text)[:]
    return None
