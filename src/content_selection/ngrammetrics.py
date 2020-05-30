from preprocessing.topic_doc_group import DocumentGroup
import numpy as np
import spacy
from content_selection.lexrank import compute_bias_vector, ir_bias, idf_weighted_vector_bias, query_sentence
import re

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
        self.query = query_sentence(document_group)

        self.bias_function = ir_bias(self.query)
        #self.bias_function = idf_weighted_vector_bias(self.query)
        #self.idf = Globals.idf.copy()
        self.unigrams, self.unigram_size, self.bigrams, self.bigram_size, self.trigrams,self.trigram_size = self.get_grams()
        self.memory = {"unigrams":self.unigrams.copy(),
                       "bigrams":self.bigrams.copy()}
        # Takes too long when used in tandem with Glob because every sentence will be considered again after reweight
        if self.config['cartesian_weight'] > 0.0:
            self.cartesian_dist, self.cart_size = self.get_cartesian_dist(document_group)
        else:
            self.cartesian_dist = None
            self.cart_size = None


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

    def _get_sentences(self, article):
        return [sentence for paragraph in article.paragraphs for sentence in paragraph.sents]

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

    def query_score(self,sentence):
        count = 0
        q = self.sent2words(self.query)
        for tok in q:
            if tok in sentence:
                count+=1
        return count/len(sentence)

    def cartesian_score(self,sentence):
        pairs = self.get_pairs(sentence)
        probas = np.array([self.cartesian_dist[pair] for pair in pairs])
        return probas.sum()/len(pairs)

    def unigram_score(self,sentence):
        unigrams = [word for word in sentence]
        probas = np.array([self.unigrams[word]  for word in unigrams])
        self.re_weight(unigrams,1)
        #idfs = np.array([self.idf[word]  for word in unigrams])
        idfs = 1
        if self.config['use_idf'] == 1:
            probas = probas*idfs
        return np.sum(probas)/len(sentence)

    def bigram_score(self,sentence):
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

    def headline_score(self,sentence,headline,lambda1,lambda2):
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

    def compute_scores(self,sentence,headline):
        if len(sentence) == 0:
            return 0.0
        scores = []
        if self.config['unigram_weight'] > 0.0:
            scores.append(self.config['unigram_weight']*self.unigram_score(sentence))
        if self.config['bigram_weight'] > 0.0:
            scores.append(self.config['bigram_weight']*self.bigram_score(sentence))
        if self.config['trigram_weight'] > 0.0:
            scores.append(self.config['trigram_weight']*self.trigram_score(sentence))
        if self.config['headline_weight'] > 0.0:
            if headline:
                scores.append(self.config['headline_weight']*self.headline_score(sentence,headline))
        if self.config['cartesian_weight'] > 0.0:
            scores.append(self.config['cartesian_weight']*self.cartesian_score(sentence))
        if self.config['query_weight'] > 0.0:
            scores.append(self.config['query_weight']*self.query_score(sentence))
        scores = np.array(scores)
        return scores.sum()

    def score(self,sentences, headline):
        return np.array([self.compute_scores(self.sent2words(sentence),headline) for sentence in sentences])

    def refresh(self):
        self.unigrams = self.memory['unigrams']
        self.bigrams = self.memory['bigrams']
        return None

    def per_article(self,articles,N,record_sents,num_batches=1.0):
        for article in articles:
            headline = article.headline
            sentences = self._get_sentences(article)
            new_sentences = []
            NUM_SENTENCES = min(N, len(sentences))
            biases = self.get_bias(sentences) * self.config['bias_weight']
            ngram_scores = (1 - self.config['bias_weight']) * self.score(sentences, headline)
            final_scores = ngram_scores + biases
            sentences_and_scores = zip(sentences,final_scores)
            sentences_and_scores = sorted(sentences_and_scores, key=lambda x:x[1],reverse=False)

            indices_to_remove = []
            for i, (sent,sc) in enumerate(sentences_and_scores):
                if self.sentence_to_be_removed(sent):
                    indices_to_remove.append(i)
            while len(sentences_and_scores) > N and len(indices_to_remove)>0:
                ind = indices_to_remove.pop(0)
                sentences_and_scores.pop(ind)
                indices_to_remove = [x-1 for x in indices_to_remove]

            sentences_and_scores = sorted(sentences_and_scores, key=lambda x:x[1],reverse=True)
            sentences_added = 0
            for tup in sentences_and_scores:
                if sentences_added == N:
                    break
                (sentence,sc) = tup
                self.re_weight2(sentence)
                if not self.sentence_to_be_removed(sentence):
                    record_sents.setdefault((sentence, article), 0.0)
                    record_sents[(sentence, article)] += sc / num_batches
                    sentences_added +=1
        return record_sents

    def sentence_to_be_removed(self, sentence):
        if sentence._.contains_quote: #quotes
            return True
        if sentence.text[-1] != '.' and sentence.text[-1] != ';': #fragments
            return True
        if re.match(r'.*\?.*',sentence.text)is not None: #questions
            return True
        if len(str(sentence).split()) < self.config['length_limit']:
            return True
        starting_pos = sentence[0].pos_
        starting_tag = sentence[0].tag_
        starting_dep = sentence[0].dep_
        second_pos = sentence[1].pos_
        second_tag = sentence[1].tag_
        second_dep = sentence[1].dep_
        if starting_pos == "DET" and starting_dep == "nsubj": #Phrases such as "That suddenly created a more serious situation at Mount St. Helens, the most active volcano in the lower 48 states."
            return True
        elif starting_pos == "ADV" and starting_dep == "nsubj":  # Phrases such as "Most had eaten cooked hot dogs the month before they became ill."
            return True
        elif starting_pos == "PRON": #Phrases like "He faces life in prison if he is convicted of murder."
            return True
        elif starting_pos == "ADV" and second_tag == "VBZ": #Phrases like "Here is the progress on a few of the important goals for the bay, based on information from the EPA's Chesapeake Bay Program."
            return True


    def basic_per_article(self,N):
        record = {}
        content = self.per_article(self.documents.articles,N,record)
        return content

    def forward_backward(self,N):
        record = {}
        record = self.per_article(self.documents.articles,N,record_sents=record,num_batches=2)
        self.refresh()
        self.documents.articles.reverse()
        record = self.per_article(self.documents.articles,N, record_sents=record,num_batches=2)
        return record

    def glob(self):
        content = []
        score_record = []
        sentences = []
        index_to_article = {}
        prev_i = 0
        for article in self.documents.articles:  # because we taking all the text irrespective of article, we should have a record of which sents came from where
            sents = self._get_sentences(article)
            for i in range(prev_i, prev_i + len(sents)):
                index_to_article[i] = article
            prev_i += len(sents)
            sentences.extend(sents)

        NUM_SENTENCES = min(self.config['num_sents_per_glob'], len(sentences))
        BIASES = self.get_bias(sentences) * self.config['bias_weight']
        remaining = [i for i in range(len(sentences))]  # remaining sentences left for selection
        index_record = []
        for n in range(NUM_SENTENCES):
            remaining_sents = [sentences[i] for i in remaining]
            ngram_scores = (1 - self.config['bias_weight']) * self.score(remaining_sents,headline=None)
            final_scores = ngram_scores
            scores = sorted([(ele, final_scores[i])
                             for i, ele in enumerate(remaining)], key=lambda x: x[1], reverse=True)
            selection = scores[0]
            sentence = sentences[selection[0]]
            self.re_weight2(sentence)
            content.append((sentence, selection[1], index_to_article[selection[0]]))
            score_record.append(selection[1])
            remaining.remove(selection[0])
            index_record.append(selection[0])
        score_record = np.array(score_record)
        score_record = score_record / score_record.sum()
        return {(tupl[0], tupl[2]): score_record[i] + BIASES[index_record[i]] for i, tupl in enumerate(content)}

    def _select_(self):
        mode = self.config['grouping']
        if mode == 'per_article':
            if self.config['forward_backward']:
                return self.forward_backward(self.config['num_sents_per_article'])
            return self.basic_per_article(self.config['num_sents_per_article'])
        elif mode == 'glob':
            return self.glob()
        return True