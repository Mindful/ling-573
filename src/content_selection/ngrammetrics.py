from preprocessing.topic_doc_group import DocumentGroup
import numpy as np
import spacy
from content_selection.lexrank import compute_bias_vector, ir_bias, idf_weighted_vector_bias, query_sentence

from common import Globals
#
# #docgroup is just docgroup
# ir_bias_function = ir_bias(query_sentence(docgroup))
# idf_weighted_vector_bias_function = idf_weighted_vector_bias(query_sentence(docgrop))
# # all_sentences is a list of spans
# bias_vector = compute_bias_vector(all_sentences, ir_bias_function)
# # OR
# bias_vector = compute_bias_vector(all_sentences, idf_weighted_vector_bias_function)



class NgramMetrics:
    def __init__(self,document_group,config):
        self.documents = document_group
        self.config = config
        self.bias_function = ir_bias(query_sentence(document_group))
        #self.idf = Globals.idf.copy()
        self.unigrams, self.unigram_size, self.bigrams, self.bigram_size, self.trigrams,self.trigram_size = self.get_grams()
        self.cartesian_dist, self.cart_size = self.get_cartesian_dist(document_group)

    def re_weight(self,data,distribution):
        if self.config['reweight_scheme'] != 'before_selection':
            return False
        elif distribution == 1:
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

    def re_weight2(self,sentence):
        if self.config['reweight_scheme'] != 'sumbasic':
            return False
        sent = self.sent2words(sentence)
        for unigram in sent:
            self.unigrams[unigram] = self.unigrams[unigram]**2
        for pair in self.get_pairs(sent):
            self.cartesian_dist[pair] = self.cartesian_dist[pair]**2
        # re-weighting for bigrams hurts scores

        return None

    def accept_token(self,token):
        tok_str = str(token).lower()
        if token.is_punct or tok_str in spacy.lang.en.stop_words.STOP_WORDS or tok_str == '`':
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

    def get_bias(self,spans):
        return compute_bias_vector(spans, self.bias_function)

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

    def get_pairs(self,sentence):
        return [word2 + ' ' + word for word in sentence for word2 in sentence if word != word2]

    def get_cartesian_dist(self,doc_group):
        record = {}
        total_pairs = 0
        for document in self.documents.articles:
            sentences, __ = self.doc2sents(document)
            for sent in sentences:
                pairs = self.get_pairs(sent)
                for pair in pairs:
                    record.setdefault(pair,0)
                    record[pair]+=1
                total_pairs += len(pairs)
        record = {k:(v/total_pairs) for k,v in record.items()}
        return record, total_pairs

    def cartesian_score(self,sentence):
        pairs = self.get_pairs(sentence)
        probas = np.array([self.cartesian_dist[pair] for pair in pairs])
        return probas.sum()/len(pairs)

    def unigram_score(self,sentence, headline):
        unigrams = [word for word in sentence]
        probas = np.array([self.unigrams[word]  for word in unigrams])
        self.re_weight(unigrams,1)
        #idfs = np.array([self.idf[word]  for word in unigrams])
        idfs = 1
        if self.config['use_idf'] == 1:
            probas = probas*idfs
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

    def mean_idf(self,sentence):
        idf_arr = np.array([Globals.idf[word] for word in self.sent2words(sentence)])
        return np.mean(idf_arr)

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

    def score(self,sentences, headline):
        scores = []
        for sentence in sentences:
            lambda1 = self.config['unigram_weight']
            lambda2 = self.config['bigram_weight']
            lambda3 = self.config['trigram_weight']
            lambda4 = self.config['headline_weight']
            sent = self.sent2words(sentence)
            if headline:
                headline_score = self.get_headline_score(sent, headline,lambda1,lambda2)
            else:
                headline_score = 0.0
            scores.append(lambda1*self.unigram_score(sent,headline) + lambda2*self.bigram_score(sent,headline)+lambda3*self.trigram_score(sent)+ lambda4*headline_score)
        scores = np.array(scores)
        return scores