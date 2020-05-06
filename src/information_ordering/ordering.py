import itertools
from operator import itemgetter
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer

class Ordering:
    def __init__(self, realization_object):
        self.selected_content = realization_object.selected_content
        self.realized_content = realization_object
        self.doc_group = realization_object.doc_group
        self.ordered_sents = self.order(realization_object)

    def order(self, realization_object):
        '''
        Given selected content, return sentences in best order

        '''
        all_sentences = realization_object.realized_content.copy()
        current_sent = self.choose_starting_sentence(all_sentences)
        ordered_sentences = [current_sent]
        all_sentences.remove(current_sent)
        remaining_sents = all_sentences

        while remaining_sents:
            next_sentence = self.select_next_sentence(current_sent, remaining_sents)
            ordered_sentences.append(next_sentence)
            remaining_sents.remove(next_sentence)
            current_sent = next_sentence

        return ordered_sentences


    def choose_starting_sentence(self, sentences):
        '''
        Ordered by article publication date, and sentence index in article secondarily
        '''
        return list(sorted(sentences, key=lambda x: (x.article.date, x.span._.sent_index)))[0]


    def select_next_sentence(self, current_sent, remaining_sents, topical_weight=0.3, 
                              succession_weight=0.3, chrono_weight=0.3):
        '''
        Determine next best sentence candidate from weighted expert scores
        '''
        scores = []

        for candidate in remaining_sents:
            topical_score = self._calculate_topical_score(current_sent, candidate)
            succession_score = self._calculate_succession_score(current_sent, candidate)
            chronological_score = self._calculate_chronological_score(current_sent, candidate)

            weighted_pref_score = (topical_score * topical_weight) \
                                  + (succession_score * succession_weight) \
                                  + (chronological_score * chrono_weight)
            scores.append((weighted_pref_score, candidate))

        return sorted(scores, key=lambda x: x[0])[-1][1]


    def _calculate_topical_score(self, current, candidate):
        return 0


    def _calculate_succession_score(self, current, candidate):
        return 0


    def _calculate_chronological_score(self, current, candidate):
        '''
        Give preference to sentences published in same article (higher to closer sentence indexes),
        then to article dates published in correct chronological sequence
        '''
        if current.article == candidate.article:
            return self._calculate_same_article_chrono_score(current.span._.sent_index, 
                                                            candidate.span._.sent_index)
        elif current.article.date <= candidate.article.date:
            return max(((current.article.date - candidate.article.date).days * 0.5) + 9, 0) \
                   + max((50 - candidate.span._.sent_index) * 0.02, 0)

        return max((current.article.date - candidate.article.date).days * -1, -7) \
               + max(((20 - candidate.span._.sent_index) * 0.05) - 1, -3)


    def _calculate_same_article_chrono_score(self, current_sent_idx, candidate_sent_idx):
        '''
        First give preference to sentences that follow the current sentence, closer higher preference
        If candidate precedes current, give preference to closer sentences
        '''
        if current_sent_idx < candidate_sent_idx:
            return max(20 - 0.2 * (candidate_sent_idx - current_sent_idx), 15)
        return 5 / (current_sent_idx - candidate_sent_idx) + 10
