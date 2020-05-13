from preprocessing import is_countworthy_token
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from collections import Counter
from common import Globals
from math import log




def pgen(generator_sentence, generatee_sentence, cluster_vocab, lmbda):
    #TODO: a lot of this should be precomputed and stuck on sentences if we're doing this
    sentence_wordcounts = Counter(token.text for token in countworthy_tokens(generator_sentence))

    word_probs = Counter({ # cluster vocab
        word: lmbda * count / sum(cluster_vocab.values()) for word, count in cluster_vocab.items()
    })
    word_probs.update({  # add sentence vocab
        word: (1-lmbda) * count / sum(sentence_wordcounts.values()) for word, count in sentence_wordcounts.items()
    })
    word_probs = { # convert to logs to avoid underflow when multiplying out
        word: log(prob) for word, prob in word_probs.items()
    }

    # add logs to avoid underflow

    generation_prob = sum( # logspace so add to multiply, multiply for exponents
        word_probs[word] * sentence_wordcounts[word] for word in generatee_sentence
    )

    #TODO: pnorm is this * 1 / sum(Counter(token.text for token in countworthy_tokens(generatee_sentence).values())









def ir_bias(query_sentence):
    query_word_counts = Counter(token._.text for token in query_sentence)

    def bias_func(sentence):
        sentence_word_counts = Counter(token._.text for token in sentence)
        return sum(
            (query_word_counts[word] + 1) * (sentence_word_counts[word] + 1) * Globals.idf[word]
            for word in query_word_counts
        )

    return bias_func


def idf_weighted_vector_bias(query_sentence):
    query_vector = idf_weighted_vector_average(query_sentence)

    def bias_func(sentence):
        return 1 - cosine_distance(query_vector, idf_weighted_vector_average(sentence))

    return bias_func


def countworthy_tokens(doc):
    return (token for token in doc if is_countworthy_token(token) and token._.text in Globals.idf)


def is_sentence_useful(sentence, vocab):
    return sum(1 for token in sentence if token._.text in vocab)


def idf_weighted_vector_average(sentence):
    tokens = [token for token in countworthy_tokens(sentence) if token.has_vector]
    idf_scores = np.array([Globals.idf[token._.text] for token in tokens])
    weighted_vectors = [
        token.vector * idf_scores[index] for index, token in enumerate(tokens)
    ]

    weighted_average = sum(weighted_vectors) / sum(idf_scores)
    return weighted_average


def tf_idf_vector_similarity_matrix(all_sentences, _):
    # normalize by idf sums, since that's what we're multiplying all the vectors by
    similarity_matrix = np.empty((len(all_sentences), len(all_sentences)))
    vectors_for_sentences = [idf_weighted_vector_average(s) for s in all_sentences]
    for i, sent_i in enumerate(all_sentences):
        for j, sent_j in enumerate(all_sentences):
            similarity_matrix[i, j] = 1 - cosine_distance(vectors_for_sentences[i], vectors_for_sentences[j])

    return similarity_matrix


def word_vector_similarity_matrix(all_sentences, _):
    similarity_matrix = np.empty((len(all_sentences), len(all_sentences)))
    for i, sent_i in enumerate(all_sentences):
        for j, sent_j in enumerate(all_sentences):
            similarity_matrix[i,j] = sent_i.similarity(sent_j)

    return similarity_matrix


def tf_idf_similarity_matrix(sentences, vocab):
    vector = dok_matrix((len(sentences), len(vocab)), dtype=np.float32)
    for index, sentence in enumerate(sentences):
        for token in sentence:
            text = token._.text
            if text in vocab:
                vector[index, vocab[text]] += Globals.idf[text]

    sentences_vector = vector.tocsr()
    similarity_matrix = cosine_similarity(sentences_vector)
    return similarity_matrix

def compute_bias_vector(all_sentences, bias_function):
    # return a vector with an entry containing the (normalized) bias for each sentences
    bias_array = np.array([bias_function(s) for s in all_sentences])
    return bias_array / bias_array.sum()


def compute_transition_matrix(threshold, similarity_matrix):
    if threshold:
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
    def __init__(self, docgroup, logger, config):
        self.config = config
        self.logger = logger
        self.docgroup = docgroup

    def __repr__(self):
        return repr(self.config)

    def rank(self):
        local_vocab = self._local_vocab(require_vector=True)
        vocab = {
            word: index for index, word in enumerate(local_vocab)
        }  # use local vocab instead of global vocab for smaller vectors; doesn't affect calculations

        sentence_counter = 0
        sentences_indices_by_article = {}
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

                else:
                    self.logger.info('LexRank skipping useless sentence "{}"'.format(sentence))

            sentences_indices_by_article[article.id] = index_list

        similarity_matrix = globals()[self.config['similarity_matrix']](all_sentences, vocab)
        transition_matrix = compute_transition_matrix(self.config['threshold'], similarity_matrix)

        bias = self.config['bias']
        if bias:
            bias_vector = compute_bias_vector(all_sentences, bias_function=globals()[bias](self.docgroup.title))
            transition_matrix = bias_and_dampen_transition_matrix(self.config['damping'], bias_vector, transition_matrix)
        else:
            transition_matrix = dampen_transition_matrix(self.config['damping'], transition_matrix)

        lexrank_vector = power_method(transition_matrix)
        results = list(zip(all_sentences, lexrank_vector))


        return results, sentences_indices_by_article

    def _local_vocab(self, require_vector=False):
        local_vocab = Counter()

        target_spans = []
        for article in self.docgroup.articles:
            for paragraph in article.paragraphs:
                for sentence in paragraph.sents:
                    target_spans.append(sentence)

        target_spans = target_spans + [
            article.headline for article in self.docgroup.articles
        ] + [self.docgroup.title, self.docgroup.narrative]

        target_spans = [span for span in target_spans if span is not None]

        for span in target_spans:
            for token in countworthy_tokens(span):
                if not require_vector or token.has_vector:
                    local_vocab[token._.text] += 1

        return local_vocab












