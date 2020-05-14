import itertools
from operator import itemgetter
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer
import numpy as np
from common import PipelineComponent


class Ordering(PipelineComponent):

    @staticmethod
    def setup():
        if Ordering.config['use_bert']:
            Ordering.model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
            Ordering.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __init__(self, realization_object):
        self.selected_content = realization_object.selected_content
        self.realized_content = realization_object
        self.doc_group = realization_object.doc_group
        self.ordered_sents = self.order(realization_object)

    def order(self, realization_object):
        '''
        Given selected content, return sentences in best order according to expert weights

        '''
        if len(realization_object.realized_content) < 1:
            raise Exception('Error', 'Received an empty list of sentences to order')

        use_BERT = Ordering.config['use_bert']
        all_sentences = realization_object.realized_content.copy()

        if use_BERT:
            succession_probs = self._calculate_BERT_succession_probs(all_sentences)
        else:
            succession_probs = np.zeros(shape=(len(all_sentences), len(all_sentences)))

        current_sent = self.choose_starting_sentence(all_sentences, succession_probs)
        ordered_sentences = [current_sent]
        all_sentences.remove(current_sent)
        remaining_sents = all_sentences

        while remaining_sents:
            next_sentence = self.select_next_sentence(current_sent, len(ordered_sentences) - 1, 
                                                      remaining_sents, succession_probs)
            ordered_sentences.append(next_sentence)
            remaining_sents.remove(next_sentence)
            current_sent = next_sentence

        return ordered_sentences


    def choose_starting_sentence(self, sentences, succession_probs):
        '''
        currently a combination of lexrank score, priviledging -1 or 0 indexed sents,
        privledging earlier article publication dates, and BERT succession scores (if use_BERT is on)

        Other options:
            only most chronological: i.e. list(sorted(sentences, key=lambda x: (x.article.date, x.span._.sent_index)))[0]
            only lexrank score: i.e. list(sorted(sentences, key=lambda x: x.score))[0]
            only BERT's succession probs: i.e. sentences[np.argmax(np.sum(succession_probs, axis=1))]
        '''
        scores = []
        earliest_date = list(sorted(sentences, key=lambda x: x.article.date))[0].article.date

        for i, sentence in enumerate(sentences):
            score = sentence.score * 100
            if sentence.span._.sent_index in (0, -1):
                score += 1
            date_penalty = (sentence.article.date - earliest_date).days * 0.1
            score -= date_penalty
            score += np.sum(succession_probs[i])
            scores.append((score, sentence))

        return sorted(scores, key=itemgetter(0))[-1][1]


    def select_next_sentence(self, current_sent, current_index, remaining_sents, succession_probs, 
                             topical_weight=0.1, succession_weight=0.3, chrono_weight=0.3):
        '''
        Determine next best sentence candidate from weighted experts for chronology, topicality, and succession
        '''
        scores = []

        for i, candidate in enumerate(remaining_sents):
            topical_score = self._calculate_topical_score(current_sent, candidate)
            succession_score = succession_probs[i][0]
            chronological_score = self._calculate_chronological_score(current_sent, candidate)
            weighted_pref_score = (topical_score * topical_weight) \
                                  + (succession_score * succession_weight) \
                                  + (chronological_score * chrono_weight)
            scores.append((weighted_pref_score, candidate))

        return sorted(scores, key=itemgetter(0))[-1][1]


    def _calculate_topical_score(self, current, candidate):
        current_sent = current.span
        candidate_sent = candidate.span

        if current_sent.has_vector and candidate_sent.has_vector:
            return current_sent.similarity(candidate_sent)
        return 0.5


    def _calculate_BERT_succession_probs(self, sentences):
        succession_probs = np.zeros(shape=(len(sentences), len(sentences)))

        for first, following in itertools.permutations(enumerate(sentences), 2):
            first_tokens = [token.text for token in first[1].span]
            following_tokens = [token.text for token in following[1].span]
            encoded = Ordering.tokenizer.encode_plus(first_tokens, text_pair=following_tokens, return_tensors='pt')
            seq_relationship_logits = Ordering.model(**encoded)[0]
            probs = softmax(seq_relationship_logits, dim=1)
            succession_probs[first[0]][following[0]] = probs[0].detach().numpy()[0]
            
        return succession_probs


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
