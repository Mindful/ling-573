from preprocessing import is_countworthy_token
from metric_computation import Metrics
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from collections import Counter

#TODO - idf should work with lemmas, everything should be lemmafied


def countworthy_tokens(doc): #TODO: update for lemmas
    return (token for token in doc if is_countworthy_token(token) and token.lower_ in Metrics.idf)


def is_sentence_useful(sentence, vocab):
    return sum(1 for token in sentence if token.lower_ in vocab)


def word_vector_similarity_matrix(all_sentences):
    similarity_matrix = np.empty((len(all_sentences), len(all_sentences)))
    for i, sent_i in enumerate(all_sentences):
        for j, sent_j in enumerate(all_sentences):
            similarity_matrix[i,j] = sent_i.similarity(sent_j)

    return similarity_matrix


def idf_weighted_vector_average(sentence):
    tokens = [token for token in countworthy_tokens(sentence) if token.has_vector]
    idf_scores = np.array([Metrics.idf[token.lower_] for token in tokens])
    weighted_vectors = [
        token.vector * idf_scores[index] for index, token in enumerate(tokens)
    ]

    weighted_average = sum(weighted_vectors) / sum(idf_scores)
    return weighted_average


def tf_idf_vector_similarity_matrix(all_sentences):
    #TODO: lemmatize
    #TODO: how to deal with things that are outside of spacy vocabulary? maybe not here...

    # normalize by idf sums, since that's what we're multiplying all the vectors by
    similarity_matrix = np.empty((len(all_sentences), len(all_sentences)))
    vectors_for_sentences = [idf_weighted_vector_average(s) for s in all_sentences]
    for i, sent_i in enumerate(all_sentences):
        for j, sent_j in enumerate(all_sentences):
            similarity_matrix[i, j] = 1 - cosine_distance(vectors_for_sentences[i], vectors_for_sentences[j])

    return similarity_matrix


def tf_idf_similarity_matrix(sentences, vocab):
    #TODO: lemmatize
    vector = dok_matrix((len(sentences), len(vocab)), dtype=np.float32)
    for index, sentence in enumerate(sentences):
        for token in sentence:
            text = token.lower_
            if text in vocab:
                vector[index, vocab[text]] += Metrics.idf[text]

    sentences_vector = vector.tocsr()
    similarity_matrix = cosine_similarity(sentences_vector)
    return similarity_matrix

def compute_bias_vector(all_sentences, bias_function):
    # return a vector with an entry containing the (normalized) bias for each sentences
    bias_array = np.array([bias_function(s) for s in all_sentences])
    return bias_array / bias_array.sum()


def compute_transition_matrix(threshold, similarity_matrix):
    if threshold is not None:
        # discrete lexrank
        adjacency_matrix = (similarity_matrix > threshold) * 1
    else:
        # continuous lexrank, degree of adjacency is similarity
        adjacency_matrix = similarity_matrix

    rowsums = adjacency_matrix.sum(axis=1, keepdims=True)
    transition_matrix = adjacency_matrix / rowsums

    return transition_matrix


def dampen_transition_matrix(damping, transition_matrix):
    n = transition_matrix.shape[0]
    damping_base = damping / n
    damping_multiplier = 1 - damping

    return damping_base + (damping_multiplier * transition_matrix)

def bias_and_dampen_transition_matrix(damping, bias_vector, transition_matrix):
    damping_multiplier = 1 - damping
    bias_base = bias_vector * damping

    return (damping_multiplier * transition_matrix) + bias_base

def power_method(matrix):
    n = matrix.shape[0]
    eigenvector = np.ones(n) / n
    matrix = matrix.transpose()

    while True:
        next_eigenvector = np.dot(matrix, eigenvector)

        if np.allclose(eigenvector, next_eigenvector):
            return next_eigenvector
        else:
            eigenvector = next_eigenvector





class LexRank:

    def disable_continous_ranking(self):
        self.threshold = None

    def __init__(self, docgroup, threshold=0.2, damping=0.15):
        self.docgroup = docgroup
        self.threshold = threshold
        self.damping = damping

    def rank(self, max_results=10):
        local_vocab = self._local_vocab(require_vector=True)
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
            if article.headline is not None:
                title = True
                sentences.append(article.headline)
            else:
                title = False

            for index, sentence in enumerate(sentences):
                if is_sentence_useful(sentence, vocab):
                    index_list.append(sentence_counter)
                    all_sentences.append(sentence)
                    sentence_counter += 1

                    if index == len(sentences) - 1 and title:
                        title_indices_by_article[article.id] = sentence_counter
                else:
                    pass
                    #print("warning: useless sentence", sentence) #TODO: replace with logging

            sentences_indices_by_article[article.id] = index_list

        #similarity_matrix = tf_idf_similarity_matrix(all_sentences, vocab)
        similarity_matrix = tf_idf_vector_similarity_matrix(all_sentences)

        transition_matrix = compute_transition_matrix(self.threshold, similarity_matrix)
        transition_matrix = dampen_transition_matrix(self.damping, transition_matrix)

        lexrank_vector = power_method(transition_matrix)
        results = list(zip(all_sentences, lexrank_vector))

        if max_results is None:
            acceptable_indices = range(0, sentence_counter)  # any index is fine, we're including everything, no max
        else:
            acceptable_indices = np.argsort(-lexrank_vector)[0:max_results]

        #TODO: bucketing and throwing away score atm, but after refactor include score and don't bucket
        #just construct Content object
        bucketed_results = {
            article_id: [results[i][0] for i in indices if i in acceptable_indices]
            for article_id, indices in sentences_indices_by_article.items()

        }

        return bucketed_results


    def _local_vocab(self, require_vector=False):
        local_vocab = set()
        for article in self.docgroup.articles:
            if article.headline is not None:
                for token in countworthy_tokens(article.headline):
                    if not require_vector or token.has_vector:
                        local_vocab.add(token.lower_)

            for paragraph in article.paragraphs:
                for token in countworthy_tokens(paragraph):
                    if not require_vector or token.has_vector:
                        local_vocab.add(token.lower_)

        return local_vocab













