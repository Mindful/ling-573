#from content_realization.sentence_similarity import is_redundant
from sentence_similarity import is_redundant
import configparser
import os.path

config_filepath = os.path.join('content_realization','config.ini')
config = configparser.ConfigParser()
config.read(config_filepath)

class Realization:

    word_quota = 100

    def __init__(self, selection_object):
        self.selected_content = selection_object
        self.doc_group = selection_object.doc_group
        self.realized_content = self.narrow_content(selection_object)


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
            text_len = len(trimmed_sentence.split())
            if text_len <= remaining_words:
                content.realized_text = trimmed_sentence
                total_words += text_len
            else:
                removed.append(content)
        return [content for content in sorted_sents if content not in removed]


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

    spans_to_remove = [] #list of tuples (start, end) of spans to remove
    for i in range(indices_to_remove):
        #Find start index
        j = i
        found_boundary = False
        while not found_boundary:
            j = j-1
            if sentence.doc[j].is_punctuation():
                continue
            else:
                j = j+1
                found_boundary = True
        start_index = j
        #Find end index
        k = i
        found_boundary = False
        while not found_boundary:
            k = k+1
            if sentence.doc[j].is_punctuation():
                continue
            else:
                k = k-1
                found_boundary = True
        end_index = k
        spans_to_remove.append((start_index,end_index))
    start_index = sentence[0].i
    output_texts = []
    for (a,b) in spans_to_remove:
        new_output = sentence.doc[start_index:a]
        new_output = new_output.text
        output_texts.append(new_output)
        start_index = b

    '''
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
    '''
    #output = ' '.join([s.text for s in output_spans])
    output = ' '.join([output_texts])
    return output

def remove_redundant_sents(content_objs):
    # will want to factor in weights to determine which sentence to remove, once weights are available
    try:
        similarity_metric = config['content_realization']['similarity_metric']
    except:
        similarity_metric = 'spacy' #default if no specification in config
    try:
        similarity_threshold = float(config['content_realization']['similarity_threshold'])
    except:
        similarity_threshold = .97 #default if no specification in config
    removed = []
    for i, content_obj in enumerate(content_objs):
        if content_obj not in removed:
            for compare_obj in content_objs[i + 1:]:
                if is_redundant(content_obj.span, compare_obj.span,
                            similarity_metric=similarity_metric,
                            similarity_threshold=similarity_threshold) \
                            and compare_obj not in removed:
                    removed.append(compare_obj)
    return [content for content in content_objs if content not in removed]

