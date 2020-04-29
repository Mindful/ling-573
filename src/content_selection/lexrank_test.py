from lexrank import STOPWORDS, LexRank
from metric_computation import Metrics


def rank_docgroup_sentences(docgroup):
    # this doc sentence list is actually useless now, feel free to ignore. probably can't delete or LexRank lib will
    # error, but we override the idf data generated rom it
    doc_sentence_text_list = [
        [sent.text for paragraph in article.paragraphs for sent in paragraph.sents] for article in docgroup.articles
    ]
    doc_sentence_text_to_article = {}
    doc_sentence_text_to_span = {}
    for article in docgroup.articles:
        for par in article.paragraphs:
            for sent in par.sents:
                doc_sentence_text_to_article[sent.text] = article.id
                doc_sentence_text_to_span[sent.text] = sent

    lxr = LexRank(doc_sentence_text_list, stopwords=STOPWORDS['en'])
    lxr.idf_score = Metrics.idf

    all_sents = [
        sent for doc in doc_sentence_text_list for sent in doc
    ]

    all_scores = lxr.rank_sentences(
        all_sents,
        threshold=0.3,
        fast_power_method=False,
    )

    candidate_sentences = lxr.get_summary(
        all_sents,
        20,
        threshold=None,
        fast_power_method=False
    )
    candidate_tuples = []
    for sent in candidate_sentences:
        candidate_tuples.append((doc_sentence_text_to_span[sent],doc_sentence_text_to_article[sent]))

    #return {'sentences':candidate_sentences,'tuples':candidate_tuples}
    return candidate_sentences
