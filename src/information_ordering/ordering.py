import dateutil
import itertools
from common import NLP

spacy_stopwords = NLP.Defaults.stop_words

class Ordering:
    def __init__(self, realization_object):
        self.selected_content = realization_object.selected_content
        self.realized_content = realization_object
        self.doc_group = realization_object.doc_group
        self.ordered_sents = self.order(realization_object)

    def order(self, realization_object):
        '''
        Given selected content, remove redundant sentences keep ordered by rank score
        '''
        return remove_redundant_sents(realization_object.realized_content)

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
        # current value (0.87) is chosen by manual inspection of ~20 sentence pairs
        # stripping down to lemmas and removing stop words did NOT seem to help i.e. nlp(" ".join([tok.lemma_ for tok in sent_1 if tok.text not in spacy_stopwords and not tok.is_punct]))
        # might consider adding comparison of doc.ents or doc.noun_chunk overlap

        # this threshold value likely needs to be tuned based on specific content selection strategy
        return sent_1.similarity(sent_2) > .97
        #return get_max_embedded_similarity(sent_1, sent_2) > .87
    return False

def get_max_embedded_similarity(sent_1, sent_2):
    '''
    ** Given two spacy spans, get similarity of sentence 2 to all subspans of sentence 1 **
    :param sent_1: spaCy span
    :param sent_2: spaCy span
    :return: max similarity score
    '''
    l1 = len(sent_1)
    l2 = len(sent_2)
    if l1 < l2:
        short_sent = sent_1
        long_sent = sent_2
    else:
        short_sent = sent_2
        long_sent = sent_1
    short_sent_len = min(l1,l2)
    long_sent_len = max(l1,l2)
    max_similarity = 0
    for i in range(0,long_sent_len-short_sent_len):
        comparison_span = long_sent[i:i+short_sent_len]
        if not (short_sent.has_vector and comparison_span.has_vector):
            continue
        sim = short_sent.similarity(comparison_span)
        if sim > max_similarity:
            max_similarity = sim
    return max_similarity

