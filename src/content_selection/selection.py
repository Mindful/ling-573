from preprocessing.topic_doc_group import DocumentGroup
from content_selection.lda import LDA
from content_selection.ngrammetrics import NgramMetrics
import spacy,re
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
                                getter=lambda token: token.lower_.title() if token.ent_type != 0 else token.lemma_)
        else:
            Token.set_extension('text',
                                getter=lambda token: token.lower_.title() if token.ent_type != 0 else token.lower_)


    def __init__(self, document_group_object, max_sentences=20):
        self.doc_group = document_group_object
        self.max_sentences = max_sentences
        self.selected_content = self.select()

    def _get_sentences(self, article):
        return [sentence for paragraph in article.paragraphs for sentence in paragraph.sents]

    def _topic_comparison(self, sentences, topic):
        num_sents = len(sentences)
        scores = {i:0 for i in range(num_sents)}
        for i in range(len(sentences)):
            for tok in sentences[i]:
                if str(tok) in topic:
                    scores[i] += 1
        return scores

    def select_lda(self):
        #TODO: rewrite for new content selection paradigm or delete
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
        METRICS = NgramMetrics(self.doc_group,Selection.config['ngram'])
        content = []
        if Selection.config['ngram']['grouping'] == 'glob':
            sentences = []
            index_to_article = {}
            prev_i = 0
            for article in self.doc_group.articles:  # because we taking all the text irrespective of article, we should have a record of which sents came from where
                sents = self._get_sentences(article)
                for i in range(prev_i,prev_i+len(sents)):
                    index_to_article[i] = article
                prev_i += len(sents)
                sentences.extend(sents)
            NUM_SENTENCES = min(Selection.config['ngram']['num_sents_per_glob'],len(sentences))
            BIASES = METRICS.get_bias(sentences) * Selection.config['ngram']['lambda5']
            remaining = [i for i in range(len(sentences))] #remaining sentences left for selection
            score_record = []
            index_record  = []
            for n in range(NUM_SENTENCES):
                remaining_sents = [ sentences[i] for i in remaining]

                """
                biases = np.array([BIASES[i] for i in remaining]) * Selection.config['ngram']['lambda5']
                ngram_scores = (1 - Selection.config['ngram']['lambda5']) * METRICS.score(remaining_sents, headline=None)
                final_scores = ngram_scores + biases
                """
                ngram_scores = (1 - Selection.config['ngram']['lambda5']) * METRICS.score(remaining_sents, headline=None)
                final_scores = ngram_scores
                scores = sorted([(ele, final_scores[i])
                                 for i,ele in enumerate(remaining)], key=lambda x: x[1], reverse=True)

                selection = scores[0]
                sentence = sentences[selection[0]]
                METRICS.re_weight2(sentence)
                content.append(Content(sentence, selection[1], index_to_article[selection[0]]))
                score_record.append(selection[1])
                remaining.remove(selection[0])
                index_record.append(selection[0])

            score_record = np.array(score_record)
            score_record = score_record/score_record.sum()
            content = [Content(c.span,score_record[i]+BIASES[index_record[i]],c.article) for i,c in enumerate(content)]

        elif Selection.config['ngram']['grouping'] == 'per_article':
            for article in self.doc_group.articles:
                headline = article.headline
                sentences = self._get_sentences(article)
                NUM_SENTENCES = min(Selection.config['ngram']['num_sents_per_article'], len(sentences))

                biases = METRICS.get_bias(sentences)*Selection.config['ngram']['lambda5']
                ngram_scores = (1-Selection.config['ngram']['lambda5'])*METRICS.score(sentences,headline)
                final_scores = ngram_scores + biases

                scores = sorted([(i,final_scores[i])  for i in range(len(sentences))],key=lambda x: x[1], reverse=True)

                selections = sorted([scores[n]
                                for n in range(NUM_SENTENCES)],
                                key=lambda x: x[0])  # get the sentence indicies in chronological order

                for tupl in selections:
                    sentence = sentences[tupl[0]]
                    score = tupl[1]
                    METRICS.re_weight2(sentence)
                    if len(str(sentence).split()) > Selection.config['ngram']['length_limit']:
                        content.append(Content(sentence,score,article))
                    else:
                        pass

        return content

    def select_simple(self):
        content = []
        for article in self.doc_group.articles:
            content.append(Content(next(article.paragraphs[0].sents), 1, article))
            content.append(Content(next(article.paragraphs[-1].sents), 1, article))

        return content

    def select_lexrank(self):
        lexrank_results, sentence_indices_by_article = LexRank(self.doc_group, Selection.logger, Selection.config['lexrank']).rank()
        articles_by_id = {article.id: article for article in self.doc_group.articles}

        content = []
        for article_id, sentence_indices in sentence_indices_by_article.items():
            for index in sentence_indices:
                content.append(Content(lexrank_results[index][0], lexrank_results[index][1], articles_by_id[article_id]))

        return content

    def select(self):
        content = sorted(self.selection_method(), key= lambda c: c.score, reverse=True)
        if self.max_sentences is not None:
            content = content[0:self.max_sentences]

        return content



