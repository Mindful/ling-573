from preprocessing.topic_doc_group import DocumentGroup
from content_selection.lda import LDA
from content_selection.ngrammetrics import NgramMetrics
import spacy, re
from content_selection.lexrank import LexRank
import metric_computation
from common import PipelineComponent, Globals
from spacy.tokens import Token
import numpy as np


class Content:
    def __init__(self, content, score, article):
        self.span = content
        self.score = score
        self.article = article
        self.realized_text = content.text

    def __repr__(self):
        return str(self.__dict__)


class Selection(PipelineComponent):
    selection_method = None

    @staticmethod
    def setup():
        Selection.selection_method = getattr(Selection, Selection.config['method'])
        if Globals.config['lemmatized_idf']:
            Token.set_extension('text',
                                getter=lambda token: token.lemma_.lower())
        else:
            Token.set_extension('text',
                                getter=lambda token: token.lower_)

    def __init__(self, document_group_object, max_sentences=20):
        self.doc_group = document_group_object
        self.max_sentences = max_sentences
        self.selected_content = self.select()

    def _get_sentences(self, article):
        return [sentence for paragraph in article.paragraphs for sentence in paragraph.sents]

    def _topic_comparison(self, sentences, topic):
        num_sents = len(sentences)
        scores = {i: 0 for i in range(num_sents)}
        for i in range(len(sentences)):
            for tok in sentences[i]:
                if str(tok) in topic:
                    scores[i] += 1
        return scores

    def select_lda(self):
        # TODO: rewrite for new content selection paradigm or delete
        raise RuntimeError("Deprecated; still performs selection per article")
        metrics = NgramMetrics(self.doc_group)

        sentences = self._get_sentences(document_group_article)
        indicies = set([])  # needs to be a set because we consider every sentence per subtopic
        subtopics = LDA(self.doc_group).subtopics
        for id in subtopics:
            scores = sorted(self._topic_comparison(sentences, subtopics[id]).items(), key=lambda x: x[1],
                            reverse=True)
            indicies.add(scores[0][0])
        selections = set([sentences[i] for i in sorted(indicies)])

        return tuple(selections)

    def select_ngram(self):
        METRICS = NgramMetrics(self.doc_group, Selection.config['ngram'])
        scores = METRICS._select_()
        return [Content(tupl[0], scores[tupl], tupl[1]) for tupl in scores]


    def select_lexrank(self):
        lexrank_results, sentence_indices_by_article = LexRank(self.doc_group, Selection.logger,
                                                               Selection.config['lexrank']).rank()
        articles_by_id = {article.id: article for article in self.doc_group.articles}

        content = []
        for article_id, sentence_indices in sentence_indices_by_article.items():
            for index in sentence_indices:
                content.append(
                    Content(lexrank_results[index][0], lexrank_results[index][1], articles_by_id[article_id]))

        return content

    def select(self):
        content = sorted(self.selection_method(), key=lambda c: c.score, reverse=True)
        if self.max_sentences is not None:
            content = content[0:self.max_sentences]

        return content

