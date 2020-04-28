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

    def __init__(self, docgroup, threshold=0.2, damping=0.15):
        self.docgroup = docgroup
        self.threshold = threshold
        self.damping = damping

    def rank(self):
        local_vocab = self._local_vocab()
        vocab = {
            word: index for index, word in enumerate(local_vocab)
        }  # TODO: this might be better as IDF vocab instead of local vocab, unsure

        sentence_counter = 0
        sentences_indices_by_article = {}
        all_sentences = []
        for article in self.docgroup.articles:
            index_list = []
            for sentence in (sent for paragraph in article.paragraphs for sent in paragraph.sents):
                index_list.append(sentence_counter)
                all_sentences.append(sentence)
                sentence_counter += 1

            sentences_indices_by_article[article.id] = index_list

        sentences_vector = self._vectorize_sentences(all_sentences, vocab)
        similarity_matrix = cosine_similarity(sentences_vector)

        transition_matrix = self._compute_transition_matrix(similarity_matrix)

        lexrank_matrix = self._power_method(transition_matrix)

        print('wiggity woo')


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

    def _local_vocab(self):
        local_vocab = set()
        for article in self.docgroup.articles:
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











