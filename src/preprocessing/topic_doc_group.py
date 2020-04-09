import re
import spacy
from spacy.tokens import Doc


def english_nlp():
    # just doing the en_core_web_lg with everything enabled for now,
    # we may want to customize and/or exclude parts of the pipeline later
    print('Loading spaCy, this may take a moment.') #TODO: replace this with logging once logging is set up
    nlp = spacy.load("en_core_web_lg")
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
    __slots__ = ['id', 'date', 'headline', 'type', 'paragraphs', 'unprocessed_paragraphs']

    def __init__(self, article):
        self.id = article.id
        self.date = article.date
        self.headline = process_text(article.headline)
        self.type = article.type
        self.paragraphs = self._process_paragraphs(article.paragraphs)
        self.unprocessed_paragraphs = article.paragraphs

    def __str__(self):
        return str({attr: getattr(self, attr) for attr in self.__slots__})

    def __repr__(self):
        return "<{} {}: {}>".format(self.__class__.__name__, self.id, self.date)

    def _process_paragraphs(self, paragraphs):
        return [nlp_parser(clean_text(p)) for p in paragraphs]


def process_text(text):
    if text:
        return nlp_parser(text)
    return None


def clean_text(text):
    # what all should happen here?

    # ideas:
    # * remove (or convert?) the location parenthetical that begins most articles, e.g. "LITTLETON, Colo. (AP) --"
    # * strange punctuation handling/removal

    # * remove spurious line breaks 
    text = re.sub("\s+\\n", " ", text)

    # * possibly remove quotes?
    # * remove taglines, e.g. "BY/ By/ Source, etc."
    return text


