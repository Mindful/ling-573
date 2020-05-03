
class Realization:

    word_quota = 100

    def __init__(self, selection_object):
        #self.ordered_sents = ordering_object.ordered_sents
        self.selected_content = selection_object
        self.doc_group = selection_object.doc_group
        self.realized_content = self.narrow_content(selection_object)
        #self.summary = self.summarize()


    def narrow_content(self,selection_object):
        '''
        Given selected content, narrow to 100 words
        :param selection_object: internal selection object from content_selection module
        :return: selection object with unwanted content removed
        '''
        original_sents = remove_redundant_sents(selection_object.selected_content)
        sorted_sents = sorted(original_sents, key=lambda x: x.score, reverse=True)
        removed = []
        total_words = 0
        for i, content in enumerate(sorted_sents):
            remaining_words = self.word_quota - total_words
            trimmed_sentence = trim(content.span)
            '''
            text_ = trimmed_sentence.text
            text_len = len(text_.split())
            '''
            text_len = len(trimmed_sentence.split())
            if text_len <= remaining_words:
                content.realized_text = trimmed_sentence
            else:
                removed.append(content)
        return [content for content in sorted_sents if content not in removed]

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
        for cand in self.ordered_sents:
            if quota_reached:
                break
            cand = trim(cand.span)
            try:
                cand = cand.text
            except:
                #cand could already be a string
                cand = cand
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


def trim(sentence):
    '''
    ** Trim sentences to reduce length and create a more focused summary **
    ** Initial Inspiration taken from CLASSY 2006 **
    :param sentence: spaCy span to be trimmed
    :return: spaCy span or string, trimmed
    TODO: try to leave everything as a spaCy span?
    '''
    ## Initial ideas for trimming (from CLASSY 2006):
    ## Extraneous Words (date lines, editor's comments, etc)
    ## Adverbs and conjunctions at start of sentence
    ## Small selection of words in the middle of sentences, like however and also (ambiguous...)
    ## Ages
    ## Relative Clause attributives (who, whom, which, when, where)
    ## Attributions (the police said) at start or end of sentence when word is not a quote.
    ####################################################
    ## Other ideas (from wes):
    ## email addresses?
    ## lines that mention the publication source ('for use by clients of the NY times service')
    ## sentences that are too short / too long (should probably be in selection).
    #####################################################
    sentence = remove_sentence_initial_terms(sentence)
    sentence = remove_appositives(sentence)
    sentence = set_initial_word_to_upper(sentence)
    try:
        sentence = nlp(sentence)
        sentence = sentence[:]
    except:
        pass
    return sentence

def set_initial_word_to_upper(sentence):
    '''
    ** given a sentence, make sure the first word is uppercase **
    :param sentence: a spaCy span or string
    :return: a string
    TODO: can I change a token to uppercase while leaving a spaCy span intact?
    '''
    try:
        sentence = sentence.text
    except:
        sentence = sentence
    sentence = sentence.capitalize()
    return sentence


def remove_sentence_initial_terms(sentence):
    '''
    ** Remove a select group of terms that occur at the start of a sentence **
    ** current list: Conjuctions, Adverbs **
    :param sentence:
    :return:
    TODO: There are some adverbs that shouldn't be removed, like when.
    '''

    ## Remove sentence-initial adverbs and conjunctions ##
    pos = sentence[0].pos_
    pos2 = sentence[0].pos
    tag = sentence[0].tag
    tag2 = sentence[0].tag_
    if pos in ('CCONJ','ADV'):
        sentence = sentence[1:]
        # Remove sentence-initial commas that might be there now
        if sentence[0].text == ',':
            sentence = sentence[1:]
    return sentence

def remove_appositives(sentence):
    '''
    ** Remove appositves from a given sentence **
    :param sentence: a spaCy span
    :return: raw text of the input span with appositives removed
    TODO: figure out how to return a span object with the target removed
    '''
    output_spans = []
    indices_to_remove = set()
    for i in range(0,len(sentence)):
        if sentence[i].dep_ == 'appos':
            indices_to_remove.add(sentence[i].idx)
            for c in sentence[i].children:
                indices_to_remove.add(c.idx)
    span_start = 0
    for i in range(0,len(sentence)):
        tok = sentence[i]
        if tok.idx in indices_to_remove:
            span_end = i
            #appositives could have commas on either side to remove as well
            if i > 0 and sentence[i-1].text == ',':
                span_end -= 1
            if span_end != span_start:
                output_spans.append(sentence[span_start:span_end])
            if i < len(sentence)-1 and sentence[i+1].text == ',':
                i = i+1 #does this work?
            span_start = i+1
        elif i == len(sentence)-1:
            output_spans.append(sentence[span_start:i+1])

    output = ' '.join([s.text for s in output_spans])
    return output

def remove_redundant_sents(content_objs):
    # will want to factor in weights to determine which sentence to remove, once weights are available
    removed = []
    for i, content_obj in enumerate(content_objs):
        if content_obj not in removed:
            for compare_obj in content_objs[i + 1:]:
                if is_redundant(content_obj.span, compare_obj.span) and compare_obj not in removed:
                    removed.append(compare_obj)
    return [content for content in content_objs if content not in removed]


def is_redundant(sent_1, sent_2):
    if sent_1.has_vector and sent_2.has_vector:
        # current value (0.87) is chosen by manual inspection of ~20 sentence pairs
        # stripping down to lemmas and removing stop words did NOT seem to help i.e. nlp(" ".join([tok.lemma_ for tok in sent_1 if tok.text not in spacy_stopwords and not tok.is_punct]))
        # might consider adding comparison of doc.ents or doc.noun_chunk overlap

        # this threshold value likely needs to be tuned based on specific content selection strategy
        return sent_1.similarity(sent_2) > .97
        #return get_max_embedded_similarity(sent_1, sent_2) > .87
    return False

def get_max_embedded_similarity(sent_1, sent_2):
    '''
    ** Given two spacy spans, get similarity of sentence 2 to all subspans of sentence 1 **
    :param sent_1: spaCy span
    :param sent_2: spaCy span
    :return: max similarity score
    '''
    l1 = len(sent_1)
    l2 = len(sent_2)
    if l1 < l2:
        short_sent = sent_1
        long_sent = sent_2
    else:
        short_sent = sent_2
        long_sent = sent_1
    short_sent_len = min(l1,l2)
    long_sent_len = max(l1,l2)
    max_similarity = 0
    for i in range(0,long_sent_len-short_sent_len):
        comparison_span = long_sent[i:i+short_sent_len]
        if not (short_sent.has_vector and comparison_span.has_vector):
            continue
        sim = short_sent.similarity(comparison_span)
        if sim > max_similarity:
            max_similarity = sim
    return max_similarity

