from preprocessing.topic_doc_group import DocumentGroup


class Selection:

    def __init__(self,document_group_object):
        self.doc_group = document_group_object
        self.selected_content = { article.id:self.select(article) for article in document_group_object.articles}

    def select(self,document_group_article):
        """
        ** Currently simplified for baseline **
        :param document_group_article: An object of class DocGroupArticle
        :return : The first sentence of the first and last paragraphs of the article
        """
        topic_sentence = None  # first sentence of first paragraph
        summary_sentence  = None  # first sentence of last paragraph
        return (topic_sentence,summary_sentence)