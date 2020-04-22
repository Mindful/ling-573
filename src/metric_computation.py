from data import load_all_articles, load_sample_articles, configure_local, DataManager, DATA_DIR
import spacy
from collections import defaultdict, Counter
import os
import pickle
from progress.bar import Bar
from math import log2


WORD_FREQ_BY_DOC = 'word_freq_by_doc.pickle'

def metric_file_name(metric_name, corpus):
    return os.path.join(DATA_DIR, corpus.name + '_' + metric_name)

def compute_word_counts_by_doc(corpus):

    tokenizer = spacy.load("en_core_web_lg", disable=['tagger', 'parser', 'ner'])

    articles = load_all_articles(corpus)
    textblob_by_id = {
        article.id: ' '.join(article.paragraphs) for article in articles
    }

    spacy_gen = ((text, article_id) for article_id, text in textblob_by_id.items())


    word_counts = defaultdict(Counter)

    bar = Bar('Processing articles...', max=len(textblob_by_id))
    for doc, article_id in tokenizer.pipe(spacy_gen, as_tuples=True):
        for token in doc:
            if not token.is_punct and not token.is_stop:
                word_counts[article_id][token.lower_] +=1
        bar.next()

    with open(metric_file_name(WORD_FREQ_BY_DOC, corpus), 'wb') as picklefile:
        pickle.dump(word_counts, picklefile)

    bar.finish()


def get_words_by_doc(corpus):
    with open(metric_file_name(WORD_FREQ_BY_DOC, corpus), 'rb') as picklefile:
        return pickle.load(picklefile)


def compute_idf(corpus):
    words_by_doc = get_words_by_doc(corpus)
    doc_count = len(words_by_doc)
    words = {word for wordlist in words_by_doc.values() for word in wordlist}
    return {
        word: log2(doc_count / sum(1 for x in words_by_doc.values() if word in x))
        for word in words
    }


if __name__ == '__main__':
    configure_local('/home/josh/clms/scrapbox/corpora') #TODO: not hardcode my local dir
    compute_word_counts_by_doc(DataManager.corpora[0])
    #compute_idf(DataManager.corpora[0])