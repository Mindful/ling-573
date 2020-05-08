from preprocessing.topic_doc_group import DocumentGroup
import numpy as np
import spacy
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS
class NgramMetrics:
    def __init__(self,document_group):
        self.documents = document_group
        self.unigrams, self.unigram_size, self.bigrams, self.bigram_size, self.trigrams,self.trigram_size = self.get_grams()

    def re_weight(self,data,distribution):
        if distribution == 1:
            for unigram in data:
                self.unigrams[unigram] = self.unigrams[unigram] - 1/self.unigram_size
                self.unigram_size -= 1
        elif distribution == 2:
            for bigram in data:
                self.bigrams[bigram] = self.bigrams[bigram]

        elif distribution == 3:
            for trigram in data:
                self.trigrams[trigram] = self.trigrams[trigram] - 1/self.trigram_size
        return True

    def accept_token(self,token):
        tok_str = str(token).lower()
        if token.is_punct or tok_str in STOP_WORDS or tok_str == '`':
            return False
        return True

    def clean_headline(self,headline):
        cleaned = None
        return cleaned

    def get_entities(self,document):
        return None

    def sent2words(self,sent):
        return [str(token).lower() for token in sent if self.accept_token(token)]

    def doc2sents(self,document):
        sentences = [self.sent2words(sentence)  for paragraph in document.paragraphs for sentence in paragraph.sents]
        sentences.append([str(document.headline).lower().split()])
        return sentences, self.get_entities(document)

    def get_grams(self):
        unigrams = {}
        bigrams = {}
        trigrams = {}
        num_unigrams = 0
        num_bigrams = 0
        num_trigrams = 0
        for document in self.documents.articles:
            sentences, ent = self.doc2sents(document)
            for s in sentences:
                num_unigrams += len(s)
                num_bigrams += (len(s)-1)
                num_trigrams += (len(s) - 2)
                for i in range(len(s)):
                    unigram = str(s[i])
                    if i > 0:
                        bigram = s[i-1] + ' ' + unigram
                        bigrams.setdefault(bigram,0)
                        bigrams[bigram]+=1
                        if i > 1:
                            trigram = s[i-2] + ' ' + bigram
                            trigrams.setdefault(trigram,0)
                            trigrams[trigram]+=1
                    unigrams.setdefault(unigram, 0)
                    unigrams[unigram] += 1

        # transform to probabilites & return
        unigrams = {unigram:(unigrams[unigram]/num_unigrams) for unigram in unigrams}
        bigrams = {bigram: (bigrams[bigram] / num_bigrams) for bigram in bigrams}
        trigrams = {trigram: (trigrams[trigram]/num_trigrams) for trigram in trigrams}
        return unigrams, num_unigrams, bigrams, num_bigrams, trigrams, num_trigrams

    def unigram_score(self,sentence, headline):
        unigrams = [word for word in sentence]
        probas = np.array([self.unigrams[word]  for word in unigrams])
        self.re_weight(unigrams,1)
        return np.sum(probas)/len(sentence)

    def bigram_score(self,sentence, headline):
        bigrams = [sentence[i-1] + ' ' + sentence[i] for i in range(1,len(sentence))]
        probas = np.array([ self.bigrams[bigram]  for bigram in bigrams])
        self.re_weight(bigrams,2)
        return np.sum(probas)/len(sentence)

    def trigram_score(self,sentence):
        trigrams = [sentence[i-2] + ' ' + sentence[i-1] + ' ' + sentence[i] for i in range(2,len(sentence))]
        probas = np.array([ self.trigrams[trigram]  for trigram in trigrams])
        self.re_weight(trigrams,3)
        return np.sum(probas)/len(sentence)

    def get_headline_score(self,sentence,headline,lambda1,lambda2):
        if headline.ents:
            points = []
            for ent in headline.ents:
                uni_sum = lambda1*np.sum(np.array([self.unigrams[word] for word in sentence if word in  str(ent).lower() ]))

                bigrams = [sentence[i - 1] + ' ' + sentence[i] for i in range(1, len(sentence))]
                bi_sum = lambda2*np.sum(np.array([ self.bigrams[bigram] for bigram in bigrams if  bigram in str(ent).lower() ]))

                points.append(uni_sum)
                points.append(bi_sum)
            return np.sum(np.array(points))
        else:
            return 0.0

    def score(self,sentence, headline, lambda1,lambda2,lambda3,lambda4):
        sent = self.sent2words(sentence)
        if headline:
            headline_score = self.get_headline_score(sent, headline,lambda1,lambda2)
        else:
            headline_score = 0.0

        return lambda1*self.unigram_score(sent,headline) + lambda2*self.bigram_score(sent,headline) +lambda3*self.trigram_score(sent)+ lambda4*headline_score