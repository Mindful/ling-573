#from content_realization.sentence_similarity import is_redundant
from sentence_similarity import is_redundant
import configparser
import os.path
import re

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
    sentence = clean_up_sentence(sentence)
    return sentence

def clean_up_sentence(sentence):
    '''
    ** Take care of little details to make sentence readable and grammatical after editing **
    :param sentence: string, the sentence for the summary
    :return: string in printable form
    '''
    ret = set_initial_word_to_upper(sentence)
    ret = remove_extra_spaces(ret)
    return ret

def remove_extra_spaces(sentence):
    '''
    ** remove extra spaces from sentence, namely final spaces before punctuation **
    :param sentence: string
    :return: string
    '''
    ret = re.sub("  +"," ",sentence)
    ret = re.sub(" +([.?!])$","\g<1>",sentence)
    return ret


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
    #sentence = sentence.capitalize()
    sentence_list = sentence.split()
    sentence_list[0] = sentence_list[0].capitalize()
    sentence = " ".join(sentence_list)
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
            #indices_to_remove.add(sentence[i].idx)
            indices_to_remove.add(sentence[i].i)
            for c in sentence[i].children:
                #indices_to_remove.add(c.idx)
                indices_to_remove.add(c.i)

    spans_to_remove = [] #list of tuples (start, end) of spans to remove
    for i in indices_to_remove:
        #Find start index
        j = i
        found_boundary = False
        while j>0:
            j = j-1
            if sentence.doc[j].is_punct:
                continue
            else:
                j = j+1
                break
        start_index = j
        #Find end index
        k = i
        found_boundary = False
        while k < len(sentence)-2: # don't consider final tok in sentence because we want sentence-final punctuation to stay
            k = k+1
            if sentence.doc[k].is_punct:
                continue
            else:
                k = k-1
                break
                #found_boundary = True
        end_index = k
        spans_to_remove.append((start_index,end_index))
    start_index = sentence[0].i
    output_texts = []
    spans_to_remove = sorted(spans_to_remove, key=lambda x:x[0])
    for (a,b) in spans_to_remove:
        new_output = sentence.doc[start_index:a]
        new_output = new_output.text
        output_texts.append(new_output)
        start_index = b+1
    if start_index <= sentence[len(sentence)-1].i+1:
        new_output = sentence.doc[start_index:sentence[len(sentence)-1].i+1]
        new_output = new_output.text
        output_texts.append(new_output)
    output = ' '.join(output_texts)
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

