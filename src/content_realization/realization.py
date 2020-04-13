
class Realization:

    word_quota = 100

    def __init__(self, ordering_object):
        self.ordered_sents = ordering_object.ordered_sents
        self.doc_group = ordering_object.doc_group
        self.summary = self.summarize()

    def summarize(self, complete_sentences = True):
        '''
        ** Takes sentences from ordering_object until 100 words reached (only full sentences). **
        ** No thought currently to order between articles **
        :param complete_sentences: True means summary will end with a complete sentence. False will fill to exactly 100 words.
        :return: summary as a list of strings.
        '''
        total_words = 0
        summary = []
        # Iterate through all spans in selection until we reach 100 words
        quota_reached = False # Marks whether we've reached word quota. Know when to exit outer loop
        for candidates in self.ordered_sents.values():
            if quota_reached:
                break
            for cand in [x.text for x in list(candidates)]:
                #cand = cand.text
                if cand in summary:
                    #avoid adding sentences that are exact duplicates
                    continue
                remaining_words = self.word_quota - total_words
                cand_len = len(cand.split()) #I think this is how word count will be measured in evaluation
                if cand_len <= remaining_words:
                    # if cand will not overfill quota, add whole span to summary
                    summary.append(cand)
                    total_words += cand_len
                else:
                    if not complete_sentences:
                        # if cand will overfill quota, take only as many words as necessary to reach quota
                        summary.append(' '.join(cand.split()[0:remaining_words]))
                        total_words += self.word_quota
                    quota_reached = True
                    break
        return summary

