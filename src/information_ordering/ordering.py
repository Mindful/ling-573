import dateutil
import itertools
import spacy

# TODO: figure out way to avoid loading English model in multiple places?
nlp = spacy.load("en_core_web_lg")
spacy_stopwords = nlp.Defaults.stop_words

class Ordering:
    def __init__(self, selection_object):
        self.selected_content = selection_object
        self.doc_group = selection_object.doc_group
        self.ordered_sents = self.order(selection_object)

    def order(self, selection_object):
        '''
        TO DO: implement ordering
        For now, chronological
        '''
        chronological = self._order_docs_chronologically(selection_object.selected_content)
        content_objs = [article[2] for article in chronological]
        flattened_content_objs = list(itertools.chain.from_iterable(content_objs))
        return remove_redundant_sents(flattened_content_objs)

    def _order_docs_chronologically(self, selected_content):
        articles_by_date = sorted([
            (parse_date_from_article_id(article_id), article_id)
            for article_id in selected_content.keys()
        ])
        return [(date, article_id, selected_content[article_id])
                for date, article_id in articles_by_date]

def remove_redundant_sents(content_objs):
    # will want to factor in weights to determine which sentence to remove, once weights are available
    removed = []
    for i, content_obj in enumerate(content_objs):
        if content_obj not in removed:
            for compare_obj in content_objs[i + 1:]:
                if is_redundant(content_obj.span, compare_obj.span) and compare_obj not in removed:
                    removed.append(compare_obj)
    return [content for content in content_objs if content not in removed]


def is_redundant(sent_1, sent_2):
    if sent_1.has_vector and sent_2.has_vector:

    # stripping down to lemmas and removing stop words did NOT seem to help i.e. nlp(" ".join([tok.lemma_ for tok in sent_1 if tok.text not in spacy_stopwords and not tok.is_punct]))
    # might consider adding comparison of doc.ents or doc.noun_chunk overlap
    # current value (0.87) is chosen by manual inspection of ~20 sentence pairs
        return sent_1.similarity(sent_2) > 0.87
    return False


def parse_date_from_article_id(article_id):
    return article_id.split('.')[0][3:]
