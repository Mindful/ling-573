from preprocessing.topic_doc_group import DocumentGroup
from content_selection.lda import LDA
from content_selection.metrics import Metrics
import spacy,re
from content_selection.lexrank import LexRank



class Content:
    def __init__(self,content,weight):
        self.span = content
        self.weight = weight


class Selection:

    selection_method = None

    def __init__(self, document_group_object):
        self.doc_group = document_group_object
        self.METRICS = Metrics(self.doc_group)
        self.selected_content = {
            article.id: [Content(span, None) for span in self.select(article)]  # will need to adjust here to add in weights correctly
            for article in document_group_object.articles
        }

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

    def select_lda(self, document_group_article):
        sentences = self._get_sentences(document_group_article)
        indicies = set([])  # needs to be a set because we consider every sentence per subtopic
        subtopics = LDA(self.doc_group).subtopics
        for id in subtopics:
            scores = sorted(self._topic_comparison(sentences, subtopics[id]).items(), key=lambda x: x[1],
                            reverse=True)
            indicies.add(scores[0][0])
        selections = set([sentences[i] for i in sorted(indicies)])

        return tuple(selections)

    def select_ngram(self, document_group_article):
        headline = document_group_article.headline
        sentences = self._get_sentences(document_group_article)
        NUM_SENTENCES = min(5, len(sentences))

        scores = sorted([(i, self.METRICS.score(sentences[i], headline, 0.4, 0.7, 0.05, 0.05))
                         for i in range(len(sentences))], key=lambda x: x[1], reverse=True)

        selections = sorted([scores[n]
                             for n in range(NUM_SENTENCES)],
                            key=lambda x: x[0])  # get the sentence indicies in chronological order

        return tuple([sentences[tupl[0]] for tupl in selections])

    def select_simple(self, document_group_article):
        num_paragraphs = len(document_group_article.paragraphs)
        return (list(document_group_article.paragraphs[0].sents)[0],
                list(document_group_article.paragraphs[num_paragraphs - 1].sents)[0]
                )

    def select_lexrank(self, document_group_article):
        #TODO: WIP
        lxr = LexRank(self.doc_group)
        lxr.rank()
        return self.select_simple(document_group_article) #TODO: replace with lexrank ranking

    def select(self,document_group_article):
        """
        :param document_group_article: An object of class DocGroupArticle
        :return : The first sentence of the first and last paragraphs of the article
        """

        return self.selection_method(document_group_article)



