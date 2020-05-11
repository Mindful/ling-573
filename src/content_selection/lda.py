from preprocessing.topic_doc_group import DocumentGroup
from sklearn.decomposition import LatentDirichletAllocation as skLDA
import spacy, numpy as np

class LDA:
    def __init__(self,document_group_object):
        self.docs = document_group_object
        self.vocab = self.get_vocab(self.docs)
        self.num_topics = 2
        self.num_words = 8
        self.subtopics = self.run()

    def accept_token(self,token):
        tok_str = str(token).lower()
        if token.is_punct or tok_str in spacy.lang.en.stop_words.STOP_WORDS or tok_str == '`':
            return False
        return True

    def get_vocab(self,documents):
        # remove punctiation and lowercase everything
        unique_words = set([  str(token).lower() for article in documents.articles
                                for paragraph in article.paragraphs
                                    for sentence in paragraph.sents
                                        for token in sentence
                                            if self.accept_token(token)])
        directory = {}
        i = 0
        for word in unique_words:
            directory[word] = i
            i+=1
        return directory

    def count(self,token_array):
        counts = {word:0 for word in token_array}
        for word in token_array:
            counts[word]+=1
        return counts

    def doc2vec(self,document):
        words = [str(token).lower() for paragraph in document.paragraphs
                                        for sentence in paragraph.sents
                                            for token in sentence
                                                if self.accept_token(token)]
        word_counts = self.count(words)
        vector = np.zeros(len(self.vocab))
        for word in word_counts:
            vector[self.vocab[word]] = word_counts[word]
        return vector


    def topN_topics(self,model, feature_names, no_top_words):
        topic_vec = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_vec[topic_idx] = " ".join([feature_names[i]for i in topic.argsort()[:-no_top_words - 1:-1]])
        return topic_vec


    def run(self):
        document_matrix = np.stack([self.doc2vec(article) for article in self.docs.articles])
        lda = skLDA(n_components=self.num_topics,n_jobs=None)
        lda.fit(document_matrix)
        topics = self.topN_topics(lda,{v:k for k,v in self.vocab.items()},10)
        return topics



