from data import load_all_articles, configure_local, DATA_DIR
from common import Globals
import spacy
import math
from collections import defaultdict, Counter
import os
from progress.bar import Bar
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse
import csv
import numpy as np
import json
from preprocessing import is_countworthy_token, clean_text


WORD_FREQ_BY_DOC = 'word_freq_by_doc'
IDF_SCORES = 'idf_scores'
IDF_META_DATA = 'idf_meta_data'
LEMMATIZED = '_lemmatized'



def metric_file_name(metric_name, corpus):
    return os.path.join(DATA_DIR, corpus.name + '_' + metric_name)


def compute_word_counts_by_doc(corpus, lemmatized=False):

    print('Loading spaCy...')
    tokenizer = spacy.load("en_core_web_lg", disable=['tagger', 'parser', 'ner'])

    articles = load_all_articles(corpus)
    textblob_by_id = {
        article.id: ' '.join(article.paragraphs) for article in articles
    }

    spacy_gen = ((clean_text(text, True), article_id) for article_id, text in textblob_by_id.items())


    word_counts = defaultdict(Counter)

    bar = Bar('Processing articles...', max=len(textblob_by_id))
    for doc, article_id in tokenizer.pipe(spacy_gen, as_tuples=True):
        for token in doc:
            if is_countworthy_token(token):
                if lemmatized:
                    word_counts[article_id][token.lemma_] += 1
                else:
                    word_counts[article_id][token.lower_] += 1
        bar.next()

    bar.finish()
    print('Vectorizing...')
    vectorizer = DictVectorizer(dtype=np.uint16)
    article_names = list(word_counts.keys())
    sparse_vector = vectorizer.fit_transform(word_counts.values())
    vocabulary = vectorizer.get_feature_names()

    metric_name = WORD_FREQ_BY_DOC + LEMMATIZED if lemmatized else WORD_FREQ_BY_DOC
    print('Saving data for sparse vector with shape', sparse_vector.shape, '...')
    scipy.sparse.save_npz(metric_file_name(metric_name+'.vector', corpus), sparse_vector)
    with open(metric_file_name(metric_name+'.articles', corpus), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(article_names)

    with open(metric_file_name(metric_name+'.vocab', corpus), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(vocabulary)

    print("Done")


def get_words_by_doc(corpus, lemmatized=False):
    metric_name = WORD_FREQ_BY_DOC + LEMMATIZED if lemmatized else WORD_FREQ_BY_DOC

    sparse_vector = scipy.sparse.load_npz(metric_file_name(metric_name+'.vector.npz', corpus))
    with open(metric_file_name(metric_name+'.vocab', corpus), newline='') as f:
        vocabulary = next(csv.reader(f))

    with open(metric_file_name(metric_name+'.articles', corpus), newline='') as f:
        articles = next(csv.reader(f))

    return sparse_vector, articles, vocabulary


def get_idf(corpus, lemmatized=False):
    if lemmatized:
        idf_scores_filename = metric_file_name(IDF_SCORES+LEMMATIZED, corpus)
        idf_meta_filename = metric_file_name(IDF_META_DATA+LEMMATIZED, corpus)
    else:
        idf_scores_filename = metric_file_name(IDF_SCORES, corpus)
        idf_meta_filename = metric_file_name(IDF_META_DATA, corpus)

    try:
        with open(idf_meta_filename) as meta_data_file:
            meta_data = json.load(meta_data_file)
            scores_array = np.load(idf_scores_filename + '.npy')
            idf_scores = defaultdict(lambda: meta_data['default_value'])
            idf_scores.update(dict(zip(meta_data['vocabulary'], scores_array)))
            return idf_scores

    except FileNotFoundError:
        try:
            print("Recomputing IDF data, please wait a moment")
            sparse_vector, _articles, vocabulary = get_words_by_doc(corpus)
            default_value, idf_scores = calculate_idf_score(sparse_vector, vocabulary, True)
            idf_data = {
                'default_value': default_value,
                'vocabulary': list(idf_scores.keys())
            }

            with open(idf_meta_filename, 'w') as outfile:
                json.dump(idf_data, outfile)
            np.save(idf_scores_filename, np.array(list(idf_scores.values())))

            return idf_scores
        except FileNotFoundError:
            print("Unable to find necessary files to compute IDF data so it will be left blank. May cause problems")
            return None


def calculate_idf_score(sparse_vector, vocabulary, smooth=False):
    vocab_indice_article_counts = Counter(sparse_vector.nonzero()[1])
    total_number_docs = sparse_vector.shape[0]

    if smooth:
        default_value = math.log(total_number_docs + 1)
    else:
        default_value = 0

    idf_scores = defaultdict(lambda: default_value)

    bar = Bar('Computing IDF scores...', max=len(vocabulary))
    for i, word in enumerate(vocabulary):
        word_doc_count = vocab_indice_article_counts[i]
        idf_scores[word] = math.log(total_number_docs / word_doc_count)
        bar.next()
    bar.finish()

    return (default_value, idf_scores)


if __name__ == '__main__':
    #configure_local('/home/josh/clms/scrapbox/corpora') #TODO: not hardcode my local dir
    print("Counting words and computing IDF (lemmatized and unlemmatized) for all corpora...")
    for corpus in Globals.corpora:
        compute_word_counts_by_doc(corpus, False)
        compute_word_counts_by_doc(corpus, True)
        get_idf(corpus, False)
        get_idf(corpus, True)
