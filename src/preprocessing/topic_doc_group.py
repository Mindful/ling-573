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


def clean_text(text):
    # what all should happen here?

    # ideas:
    # * remove (or convert?) the location parenthetical that begins most articles, e.g. "LITTLETON, Colo. (AP) --"
    text = re.sub(r'^.{0,50}\(AP\) (--)*', '', text) # remove (AP) -- and previous text for anything up to 50 chars from beginning of line
    text = re.sub(r'^[A-Z | \w |,]*_', '', text) # remove location and underscore, e.g. "NEW YORK _"

    # taglines, websites, etc.
    text = re.sub(r'^\w*on the net.*$', '', text, flags=re.IGNORECASE) # remove 'on the net' and everything following
    text = re.sub(r'^https?://\S+', '', text) # remove urls
    text = re.sub(r'\w*e-?mail.{0,100}', '', text, flags=re.IGNORECASE)  # remove email and following up to 100 chars, e.g. E-mail: triggp(at)nytimes.com.
    text = re.sub(r'^\w*story filed by.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. Story Filed By Cox Newspapers
    text = re.sub(r'^\w*for use by.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. For Use By Clients of the New York Times News Service
    text = re.sub(r'^\w*photos and graphics.{0,100}', '', text, flags=re.IGNORECASE)  # PHOTOS AND GRAPHICS:
    text = re.sub(r'\w*phone:.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. Phone: (888) 603-1036
    text = re.sub(r'\w*pager:.{0,100}', '', text, flags=re.IGNORECASE)  # remove Pager: (800) 946-4645 (PIN 599-4539).
    text = re.sub(r'^\w*technical problems.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. TECHNICAL PROBLEMS:   Peter Trigg



    # * remove spurious line breaks 
    text = re.sub('\s+\\n', ' ', text)
    text = re.sub('(\S+)\\n', r'\1 ', text)


    # * possibly remove quotes?
    # * remove taglines, e.g. "BY/ By/ Source, etc."
    return text.strip()


