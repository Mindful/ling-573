


class Ordering:
    def __init__(self, realization_object):
        self.selected_content = realization_object.selected_content
        self.realized_content = realization_object
        self.doc_group = realization_object.doc_group
        self.ordered_sents = self.order(realization_object)

    def order(self, realization_object):
        '''
        Given selected content, return sentences in best order
        '''
        return [content for content in realization_object.realized_content]

