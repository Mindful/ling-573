import dateutil
import itertools


class Ordering:
    def __init__(self, selection_object):
        self.selected_content = selection_object
        self.doc_group = selection_object.doc_group
        self.ordered_sents = self.order(selection_object)

    def order(self, selection_object):
        '''
        TO DO: implement ordering
        For now, chronological
        '''
        chronological = self._order_docs_chronologically(selection_object.selected_content)
        sents = [article[2] for article in chronological]
        return list(itertools.chain.from_iterable(sents))

    def _order_docs_chronologically(self, selected_content):
        articles_by_date = sorted([
            (parse_date_from_article_id(article_id), article_id)
            for article_id in selected_content.keys()
        ])
        return [(date, article_id, selected_content[article_id])
                for date, article_id in articles_by_date]


def parse_date_from_article_id(article_id):
    return article_id.split('.')[0][3:]
