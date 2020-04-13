
class Ordering:
    def __init__(self, selection_object):
        self.selected_content = selection_object.selected_content
        self.doc_group = selection_object.doc_group
        self.ordered_sents = self.order(selection_object)

    def order(self, selection_object):
        '''
        TO DO: implement ordering
        '''
        return selection_object.selected_content

