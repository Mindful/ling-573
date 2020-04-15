from preprocessing.topic_doc_group import DocumentGroup
from content_selection.lda import LDA
import spacy,re



class Content:
    def __init__(self,content,weight):
        self.span = content
        self.weight = weight

class Selection:

    def __init__(self,document_group_object):
        self.doc_group = document_group_object
        self.selected_content = {
            article.id: [Content(span, None) for span in self.select(article)]  # will need to adjust here to add in weights correctly
            for article in document_group_object.articles
        }
        #self.subtopics = self.lda()


    def select(self,document_group_article):
        """
        ** Currently simplified for baseline **
        :param document_group_article: An object of class DocGroupArticle
        :return : The first sentence of the first and last paragraphs of the article
        """

        num_paragraphs = len(document_group_article.paragraphs)
        return ( list(document_group_article.paragraphs[0].sents)[0],
                 list(document_group_article.paragraphs[num_paragraphs-1].sents)[0]
               )
    def lda(self):
        return LDA(self.doc_group)

