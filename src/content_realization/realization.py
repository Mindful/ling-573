from content_selection.selection import Selection
from content_realization.utils import span_len, word_span

WORD_QUOTA = 100

class Realization:
    def __init__(self, selection_object):
        self.selection = selection_object
        self.word_quota = WORD_QUOTA
        self.summary = self.summarize()

    def summarize(self):
        total_words = 0
        summary = ""
        # Iterate through all spans in selection until we reach 100 words
        quota_reached = False # Marks whether we've reached word quota. Know when to exit outer loop
        for candidates in self.selection.selected_content.values():
            if quota_reached:
                break
            for cand in list(candidates):
                remaining_words = self.word_quota - total_words
                cand_len = span_len(cand)
                if cand_len <= remaining_words:
                    # if cand will not overfill quota, add whole span to summary
                    summary = summary+cand.text+"\n"
                    total_words += cand_len
                else:
                    # if cand will overfill quota, take only as many words as necessary to reach quota
                    new_span = word_span(cand.doc, cand.start, cand.start+remaining_words)
                    summary = summary + new_span.text
                    total_words += self.word_quota
                    quota_reached = True
                    break
        return summary
