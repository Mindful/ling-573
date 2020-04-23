from preprocessing.topic_doc_group import DocumentGroup
import numpy as np
import spacy
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS
class Metrics:
    def __init__(self,document_group):
        self.documents = document_group
        self.unigrams, self.bigrams = self.get_grams()


    def accept_token(self,token):
        tok_str = str(token).lower()
        if token.is_punct or tok_str in STOP_WORDS or tok_str == '`':
            return False
        return True


    def get_entities(self,document):
        return None

    def doc2sents(self,document):
        sentences = [[str(token).lower() for paragraph in document.paragraphs
                                        for sentence in paragraph.sents
                                            for token in sentence
                                                if self.accept_token(token)]]

        return sentences, self.get_entities(document)




    def get_grams(self):
        unigrams = {}
        bigrams = {}
        num_unigrams = 0
        num_bigrams = 0
        for document in self.documents:
            sentences = self.doc2sents(document)
            for s in sentences:
                num_unigrams += len(s)
                num_bigrams += (len(s)-1)
                for i in range(len(s)):
                    unigram = s[i]
                    if i > 0:
                        bigram = s[i-1] + ' ' + unigram
                        bigrams.setdefault(bigram,0)
                        bigrams[bigram]+=1

                    unigrams.setdefault(unigram, 0)
                    unigrams[unigram] += 1

        # transform to probabilites & return
        unigrams = {unigram:(unigrams[unigram]/num_unigrams) for unigram in unigrams}
        bigrams = {bigram: (bigrams[bigram] / num_bigrams) for bigram in bigrams}
        return unigrams, bigrams