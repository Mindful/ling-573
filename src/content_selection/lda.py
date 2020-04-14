from preprocessing.topic_doc_group import DocumentGroup
from sklearn.decomposition import LatentDirichletAllocation as skLDA
import spacy, numpy as np
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

class LDA:
    def __init__(self,document_group_object):
        self.docs = document_group_object
        self.vocab = self.get_vocab(self.docs)
        self.num_topics = 5
        self.num_words = 10
        self.run()

    def get_vocab(self,documents):
        # remove punctiation and lowercase everything
        unique_words = [  str(token).lower() for article in documents.articles
                                for paragraph in article.paragraphs
                                    for sentence in paragraph.sents
                                        for token in sentence
                                            if not token.is_punct and not token in STOP_WORDS]
        return { unique_words[i]:i for i in range(len(unique_words)) }

    def count(self,token_array):
        counts = {word:0 for word in token_array}
        for word in token_array:
            counts[word]+=1
        return counts

    def doc2vec(self,document):
        words = [str(token).lower() for paragraph in document.paragraphs
                                for sentence in paragraph.sents
                                    for token in sentence
                                        if not token.is_punct and not token in STOP_WORDS]
        word_counts = self.count(words)
        vector = np.zeros(len(self.vocab))
        for word in word_counts:
            vector[self.vocab[word]] = word_counts[word]
        return vector



    def run(self):
        documents = np.stack([self.doc2vec(article) for article in self.docs.articles])
        lda = skLDA(n_components=self.num_topics,n_jobs=-1)
        lda.fit(self.count())
        print(lda)
