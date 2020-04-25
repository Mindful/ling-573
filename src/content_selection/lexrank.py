from preprocessing import is_countworthy_token
from metric_computation import Metrics
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class LexRank:

    def __init__(self, docgroup, threshold=0.2, damping=0.15):
        self.docgroup = docgroup
        self.threshold = threshold


    def continuous(self):
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
        adjacency_matrix = (similarity_matrix > self.threshold) * 1
        rowsums = np.sum(adjacency_matrix, axis=1)
        transition_matrix = adjacency_matrix / rowsums[:, None] # "B" from the paper

        print('wiggity')


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











