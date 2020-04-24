from preprocessing.topic_doc_group import DocumentGroup
from content_selection.lda import LDA
from content_selection.metrics import Metrics
import spacy,re



class Content:
    def __init__(self,content,weight):
        self.span = content
        self.weight = weight

class Selection:

    def __init__(self,document_group_object,use_lda=False,use_ngram=True):
        self.doc_group = document_group_object
        self.USE_LDA = use_lda
        self.USE_NGRAM = use_ngram
        self.subtopics = self.lda()
        self.METRICS = Metrics(self.doc_group)
        self.selected_content = {
            article.id: [Content(span, None) for span in self.select(article)]  # will need to adjust here to add in weights correctly
            for article in document_group_object.articles
        }

    def get_sentences(self,article):
        return [ sentence for i in range(len(article.paragraphs)) for sentence in list(article.paragraphs[i].sents) ]


    def topic_comparison(self,sentences, topic):
        num_sents = len(sentences)
        scores = {i:0 for i in range(num_sents)}
        for i in range(len(sentences)):
            for tok in sentences[i]:
                if str(tok) in topic:
                    scores[i]+=1
        return scores


    def select(self,document_group_article):
        """
        ** Currently simplified for baseline **
        :param document_group_article: An object of class DocGroupArticle
        :return : The first sentence of the first and last paragraphs of the article
        """
        if self.USE_LDA:
            sentences = self.get_sentences(document_group_article)
            indicies = set([]) # needs to be a set because we consider every sentence per subtopic
            for id in self.subtopics:
                scores = sorted(self.topic_comparison(sentences,self.subtopics[id]).items(),key=lambda x:x[1],reverse=True)
                #selections.add(sentences[scores[0][0]])
                indicies.add(scores[0][0])
            selections = set([sentences[i] for i in sorted(indicies)])

            return tuple(selections)

        elif self.USE_NGRAM:
            headline = document_group_article.headline
            sentences = self.get_sentences(document_group_article)
            NUM_SENTENCES = min(2,len(sentences))

            scores = sorted([(i,self.METRICS.score(sentences[i],headline,0.3,0.7,0.05)) for i in range(len(sentences))],key=lambda x:x[1],reverse=True)
            selections = sorted( [ scores[n] for n in range(NUM_SENTENCES) ],key=lambda x:x[0])  # get the sentence indicies in chronological order



            return tuple([ sentences[tupl[0]]for tupl in selections])

        else:
            num_paragraphs = len(document_group_article.paragraphs)
            return ( list(document_group_article.paragraphs[0].sents)[0],
                    list(document_group_article.paragraphs[num_paragraphs-1].sents)[0]
                   )

    def lda(self):
        return LDA(self.doc_group).subtopics

