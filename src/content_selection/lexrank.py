from preprocessing import is_countworthy_token
from metric_computation import Metrics
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class LexRank:

    def disable_damping(self):
        self.damping = None

    def disable_continous_ranking(self):
        self.threshold = None

    def __init__(self, docgroup, threshold=0.2, damping=0.15, include_headlines=True, bias_for_headlines=True):
        self.docgroup = docgroup
        self.threshold = threshold
        self.damping = damping
        self.include_headlines = include_headlines
        self.bias_for_headlines = bias_for_headlines

    def rank(self):
        local_vocab = self._local_vocab()
        vocab = {
            word: index for index, word in enumerate(local_vocab)
        }  # use local vocab instead of global vocab for smaller vectors; doesn't affect calculations

        sentence_counter = 0
        sentences_indices_by_article = {}
        title_indices_by_article = {}
        all_sentences = []
        for article in self.docgroup.articles:
            index_list = []
            sentences = [sent for paragraph in article.paragraphs for sent in paragraph.sents]
            if self.include_headlines and article.headline is not None:
                title = True
                sentences.append(article.headline)
            else:
                title = False

            for index, sentence in enumerate(sentences):
                if self._is_sentence_useful(sentence, vocab):
                    index_list.append(sentence_counter)
                    all_sentences.append(sentence)
                    sentence_counter += 1

                    if index == len(sentences) - 1 and title:
                        title_indices_by_article[article.id] = sentence_counter
                else:
                    print("warning: useless sentence", sentence) #TODO: replace with logging

            sentences_indices_by_article[article.id] = index_list

        sentences_vector = self._vectorize_sentences(all_sentences, vocab)
        similarity_matrix = cosine_similarity(sentences_vector)

        transition_matrix = self._compute_transition_matrix(similarity_matrix)

        lexrank_matrix = self._power_method(transition_matrix)

        return sorted(zip(all_sentences, lexrank_matrix), key=lambda x: x[1], reverse=True)

    def _is_sentence_useful(self, sentence, vocab):
        return sum(1 for token in sentence if token.lower_ in vocab)



    def _dampen_transition_matrix(self, transition_matrix):
        n = transition_matrix.shape[0]
        damping_base = self.damping / n
        damping_multiplier = 1 - self.damping

        return damping_base + (damping_multiplier * transition_matrix)


    def _compute_transition_matrix(self, similarity_matrix):
        if self.threshold is not None:
            # discrete lexrank
            adjacency_matrix = (similarity_matrix > self.threshold) * 1
        else:
            # continuous lexrank, degree of adjacency is similarity
            adjacency_matrix = similarity_matrix

        rowsums = adjacency_matrix.sum(axis=1, keepdims=True)
        transition_matrix = adjacency_matrix / rowsums
        if self.damping is not None:
            return self._dampen_transition_matrix(transition_matrix)
        else:
            return transition_matrix

    def _power_method(self, matrix):
        #TODO: this method isn't working properly
        n = matrix.shape[0]
        eigenvector = np.ones(n) / n
        matrix = matrix.transpose()

        while True:
            next_eigenvector = np.dot(matrix, eigenvector)

            if np.allclose(eigenvector, next_eigenvector):
                return next_eigenvector
            else:
                eigenvector = next_eigenvector

    def _local_vocab(self):
        local_vocab = set()
        for article in self.docgroup.articles:
            if self.include_headlines and article.headline is not None:
                for token in article.headline:
                    if is_countworthy_token(token):
                        local_vocab.add(token.lower_)

            for paragraph in article.paragraphs:
                for token in paragraph:
                    if is_countworthy_token(token):
                        local_vocab.add(token.lower_)

        return local_vocab

    def _vectorize_sentences(self, sentences, vocab):
        vector = dok_matrix((len(sentences), len(vocab)), dtype=np.float32)
        for index, sentence in enumerate(sentences):
            for token in sentence:
                text = token.lower_
                if text in vocab:
                    vector[index, vocab[text]] += Metrics.idf[text]

        return vector.tocsr()











