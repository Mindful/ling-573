from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch
from common import PipelineComponent, ROOT_DIR
from os.path import join
import datetime

WORD_QUOTA = 100

class Realization(PipelineComponent):
    @staticmethod
    def setup():
        if Realization.config['similarity_metric'] == 'bert':
            Realization.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
            Realization.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")


    def __init__(self, selection_object):
        self.selected_content = selection_object
        self.doc_group = selection_object.doc_group
        self.input_words = get_num_words_in_collection(self.selected_content.selected_content)
        self.realized_content = self.narrow_content(selection_object)

        self.log_realization_changes()


    def log_realization_changes(self):
        for content_object in self.realized_content:
            Realization.logger.debug("Article ID: {}".format(content_object.article.id))
            Realization.logger.debug("\tInput Text: {}".format(content_object.span.text))
            Realization.logger.debug("\tOutput Text: {}".format(content_object.realized_text))

        initial_word_count = get_num_words_in_collection(self.selected_content.selected_content)
        Realization.logger.debug("\tNarrowed from {} words to {} words".format(initial_word_count, get_num_words_in_collection(self.realized_content)))
        Realization.logger.debug("\tNarrowed from {} sentences to {} sentences".format(len(self.selected_content.selected_content), len(self.realized_content)))


    def narrow_content(self, selection_object):
        '''
        Given selected content, narrow to 100 words
        :param selection_object: internal selection object from content_selection module
        :return: selection object with unwanted content removed
        '''
        cand_sents = sorted(selection_object.selected_content, key=lambda x: x.score, reverse=True)

        ## First apply methods that leave the spaCy span intact (add or remove full content objects)
        if Realization.config['remove_quotes']:
            cand_sents = remove_quotes(cand_sents)
        if Realization.config['remove_questions']:
            cand_sents = remove_questions(cand_sents)
        if Realization.config['remove_sentences_starting_with_pronouns']:
            cand_sents = remove_sentences_starting_with_pronouns(cand_sents)
        if Realization.config['remove_subjectless_sentences']:
            cand_sents = remove_subjectless_sentences(cand_sents)
        if len(Realization.config['remove_full_spans_that_match']) > 0:
            cand_sents = filter_content_by_regex_list(cand_sents, Realization.config['remove_full_spans_that_match'])

        cand_sents = remove_fragments(cand_sents)

        ## trim_content_objs modifies the realized_text on each content object. span no longer reliable
        cand_sents = trim_content_objs(cand_sents)

        if Realization.config['remove_attributions']:
            cand_sents = remove_attributions(cand_sents)

        if len(Realization.config['remove_subspans_that_match']) > 0:
            cand_sents = remove_text_by_regex_list(cand_sents, Realization.config['remove_subspans_that_match'])

        unique_sents, overflow_sents = remove_redundant_sents(cand_sents)
        unique_sents = remove_under_min_length(unique_sents, Realization.config['minimum_sentence_length'])
        unique_sents = clean_up_objects(unique_sents)

        total_words = 0
        unique_in_range_sents = []

        for sent in unique_sents:
            sent_word_count = content_obj_word_count(sent)
            if total_words + sent_word_count <= WORD_QUOTA:
                unique_in_range_sents.append(sent)
                total_words += sent_word_count
        return use_extra_quota_space(unique_in_range_sents, overflow_sents, selection_object.selected_content)


def get_num_words_in_collection(content_objs):
    word_counts = [content_obj_word_count(content)for content in content_objs]
    return sum(word_counts)


def content_obj_word_count(content_obj):
    return len(content_obj.realized_text.split())


def remove_under_min_length(content_objs, min_length):
    under_min_length = lambda x: content_obj_word_count(x) <= min_length
    return removal_step(under_min_length, content_objs, 'remove_under_min_length')


def remove_questions(content_objs):
    is_question = lambda x: re.match(r".*\?.*", x.realized_text) is not None
    return removal_step(is_question, content_objs, 'remove_questions')


def remove_quotes(content_objs):
    is_quote = lambda x: x.span._.contains_quote
    return removal_step(is_quote, content_objs, 'remove_quotes')


def remove_fragments(content_objs):
    is_frag = lambda x: x.realized_text[-1] != '.' and x.realized_text[-1] != ';'
    return removal_step(is_frag, content_objs, 'remove_fragments')


def removal_step(removal_function, content_objs, label):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, label):
        return content_objs

    words_trimmed = 0
    removed = []

    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            break
        if removal_function(content_obj):
            removed.append(content_obj)
            words_trimmed += content_obj_word_count(content_obj)
            Realization.logger.debug('{}: removing sentence: {}.'.format(label, content_obj.realized_text))

    return [content for content in content_objs if content not in removed]


def no_extra_remaining(trimming_quota, category):
    if trimming_quota <= 0:
        Realization.logger.debug('{}: already hit trimming limit. skip method.'.format(category))
        return True
    return False


def trim_content_objs(content_objs):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, 'trim_content_objs'):
        return content_objs

    words_trimmed = 0
    new_content_objs = []
    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            Realization.logger.debug("new_content_objs: hit trimming limit. skip rest of method.")
            new_content_objs.append(content_obj)
            continue
        new_text = trim(content_obj.span)
        content_obj.realized_text = new_text
        new_content_objs.append(content_obj)
    return new_content_objs


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
    if Realization.config['remove_sentence_initial_terms']:
        sentence = remove_sentence_initial_terms(sentence)
    if Realization.config['remove_appositives']:
        sentence_text = remove_appositives(sentence)
    else:
        sentence_text = sentence.text

    sentence_text = handle_double_dash(sentence_text)
    sentence_text = handle_initial_punct(sentence_text)
    return sentence_text


def handle_double_dash(text):
    text = re.sub(r'\s+--\s+', r'--', text)
    return re.sub(r'^-- ', '', text)


def handle_initial_punct(text):
    text = re.sub(r'^- ', r'', text)
    text = re.sub(r'^; ', r'', text)
    return re.sub(r'^A: ', '', text)


def clean_up_objects(content_objs):
    new_content_objs = []
    for content_obj in content_objs:
        content_obj.realized_text = clean_up_sentence(content_obj.realized_text)
        new_content_objs.append(content_obj)
    return new_content_objs


def clean_up_sentence(sentence):
    '''
    ** Take care of little details to make sentence readable and grammatical after editing **
    :param sentence: string, the sentence for the summary
    :return: string in printable form
    '''
    text = set_initial_word_to_upper(sentence)
    return remove_extra_spaces(text)


def remove_extra_spaces(sentence):
    '''
    ** remove extra spaces from sentence, namely final spaces before punctuation **
    :param sentence: string
    :return: string
    '''
    text = re.sub("  +", " ", sentence)
    return re.sub(r" +([.?!,])", "\g<1>", text)


def set_initial_word_to_upper(sentence):
    '''
    ** given a sentence, make sure the first word is uppercase **
    :param sentence: string
    :return: a string
    TODO: can I change a token to uppercase while leaving a spaCy span intact?
    '''
    sentence_list = sentence.split()
    sentence_list[0] = sentence_list[0].capitalize()
    return " ".join(sentence_list)


def remove_sentence_initial_terms(sentence):
    '''
    ** Remove a select group of terms that occur at the start of a sentence **
    ** current list: Conjuctions, Adverbs **
    :param sentence:
    :return:
    TODO: There are some adverbs that shouldn't be removed, like when.
    '''
    ## Remove sentence-initial adverbs and conjunctions ##
    finished = False
    while not finished:
        tag = sentence[0].tag_
        text = sentence[0].text
        exceptions = ('Now', 'Soon', 'Most')

        if tag in ('CC','RB','RBS','RBR') and text not in exceptions:
            sentence = sentence[1:]
            # Remove sentence-initial commas that might be there now
            if sentence[0].text == ',':
                sentence = sentence[1:]
        else:
            finished = True
    return sentence


def remove_appositives(sentence):
    '''
    ** Remove appositives from a given sentence **
    :param sentence: a spaCy span
    :return: raw text of the input span with appositives removed
    TODO: figure out how to return a span object with the target removed
    '''
    indices_to_remove = set()
    for i in range(0,len(sentence)):
        if sentence[i].dep_ == 'appos':
            indices_to_remove.add(sentence[i].i)
    # get all children of all appositives
    appositive_indices = list(indices_to_remove)
    while len(appositive_indices) !=0:
        ind = appositive_indices[0]
        tok = sentence.doc[ind]
        for c in tok.children:
            appositive_indices.append(c.i)
            indices_to_remove.add(c.i)
        appositive_indices.pop(0)

    spans_to_remove = [] #list of tuples (start, end) of spans to remove
    for i in indices_to_remove:
        #Find start index
        j = i
        found_boundary = False
        while j>0:
            j = j-1
            if sentence.doc[j].is_punct or sentence.doc[j].tag_ in ('CC','RB'):
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
    sim_metric = Realization.config['similarity_metric']
    sim_threshold = Realization.config['similarity_threshold']
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, 'remove_redundant_sents'):
        return (content_objs, [])

    words_trimmed = 0
    removed = []

    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            break

        if content_obj not in removed:
            for compare_obj in content_objs:
                if words_trimmed >= stop_after_trimming:
                    break

                if not compare_obj is content_obj:
                    if is_redundant(content_obj, compare_obj, sim_metric, sim_threshold) and compare_obj not in removed:
                        removed.append(compare_obj)
                        words_trimmed += content_obj_word_count(compare_obj)

                        Realization.logger.debug("Removing redundant sentence")
                        Realization.logger.debug("Kept sentence: " + content_obj.span.text)
                        Realization.logger.debug("Removed sentence: " + compare_obj.span.text)

    return ([content for content in content_objs if content not in removed], removed)


def is_redundant(sent_1, sent_2, similarity_metric='spacy', similarity_threshold=.97):
    '''
    ** Given two sentences, check if they're redundant
    :param sent_1: a spacy span built from a sentence
    :param sent_2: a second spacy span built from another sentence
    :param similarity_metric: what similarity measure to use. 'spacy' or 'bert'
    :param similarity_threshold: what % similarity is the cutoff for redundancy
    :return: True/False for redundant/not redundant
    '''
    if redundant_lexical_overlap(sent_1, sent_2, 0.74):
        return True

    if similarity_metric=='bert':
        sim = bert_similarity(sent_1.span, sent_2.span)
    else:
        sim = spacy_similarity(sent_1.span, sent_2.span)
    return sim > similarity_threshold


def redundant_lexical_overlap(sent_1, sent_2, threshold):
    sent_1_tokens = [tok.lower() for tok in re.split(" |-", sent_1.realized_text[:-1])]
    sent_2_tokens = [tok.lower() for tok in re.split(" |-", sent_2.realized_text[:-1])]
    token_overlap = list(set(sent_1_tokens) & set(sent_2_tokens))

    if len(token_overlap) / len(sent_2_tokens) > threshold:
        return True
    return False


def spacy_similarity(sent_1, sent_2):
    if sent_1.has_vector and sent_2.has_vector:
        return sent_1.similarity(sent_2)
    return False


def bert_similarity(sent_1, sent_2):
    '''
    ** Get similarity measure using BERT finetuned on mrpc **
    ** Taken mostly from sample code from huggingfac **
    :param sentence_1: spacy span for sentence 1
    :param sentence_2: spacy span for sentence 2
    :return: % likelihood that sentences are paraphrases
    '''

    sequence_0 = sent_1.text
    sequence_1 = sent_2.text
    possible_paraphrase = Realization.tokenizer.encode_plus(sequence_0,sequence_1,return_tensors="pt")

    paraphrase_classification_logits = Realization.model(**possible_paraphrase)[0]
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0][1] #[1] is the class for "is paraphrase"
    return paraphrase_results


def filter_content_by_regex_list(content_objs,regex_list,max_length = 100):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - max_length

    if no_extra_remaining(stop_after_trimming, 'filter_content_by_regex_list'):
        return content_objs

    words_trimmed = 0
    removed = []
    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            Realization.logger.debug("filter_content_by_regex_list: hit trimming limit. skip rest of method.")
            break
        for regex in regex_list:
            if re.match(regex,content_obj.span.text) is not None:
                removed.append(content_obj)
                words_trimmed += content_obj_word_count(content_obj)
                Realization.logger.debug("Removing sentence with matching regex: "+ content_obj.span.text)
                break
    return [content for content in content_objs if content not in removed]


def remove_text_by_regex_list(content_objs,regex_list,max_length = 100):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - max_length

    if no_extra_remaining(stop_after_trimming, 'remove_text_by_regex_list'):
        return content_objs

    words_trimmed = 0
    new_content_objs = []
    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            Realization.logger.debug("remove_text_by_regex_list: hit trimming limit. skip rest of method.")
            new_content_objs.append(content_objs)
            continue
        for regex in regex_list:
            if re.search(regex,content_obj.realized_text) is not None:
                new_text = re.sub(regex,"",content_obj.realized_text)
                Realization.logger.debug("Changing text: -- " +
                                        content_obj.span.text+" -- to "+
                                        new_text)
                content_obj.realized_text = new_text
        new_content_objs.append(content_obj)
    if words_trimmed >= stop_after_trimming:
        Realization.logger.debug("remove_text_by_regex_list: hit trimming limit. skip rest of method.")
    return new_content_objs


def remove_subjectless_sentences(content_objs,max_length=100):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - max_length
    if stop_after_trimming <= 0:
        Realization.logger.debug("remove_subjectless_sentences: already below word limit. skip method.")
        return content_objs
    words_trimmed = 0
    removed = []
    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            Realization.logger.debug("remove_subjectless_sentences: hit trimming limit. skip rest of method.")
            break
        found_subj = False
        for tok in content_obj.span:
            if tok.dep_ in ('nsubj','nsubjpass'):
                found_subj = True
                break
        if found_subj == False:
            removed.append(content_obj)
            Realization.logger.debug("Removing sentence with no subject: "+
                                        content_obj.span.text)
    return [content for content in content_objs if content not in removed]


def remove_attributions(content_objs):
    return [remove_attribution(content) for content in content_objs]


def remove_attribution(content_obj):
    text = content_obj.realized_text
    # sentence final
    text = re.sub(r', citing [\s|a-zA-Z]+\.$', '.', text, flags=re.IGNORECASE)
    text = re.sub(r', [a-zA-Z]* said\.$', '.', text)
    text = re.sub(r', [a-zA-Z]*\s([a-zA-Z]*\s)?([a-zA-Z]*\s)?said\.$', '.', text)
    text = re.sub(r', [a-zA-Z]* said today\.$', '.', text)
    text = re.sub(r', said [a-zA-Z]*\.$', '.', text)
    text = re.sub(r', [a-zA-Z]* press reported today\.$', '.', text)
    text = re.sub(r', [a-zA-Z]* reported\.$', '.', text)
    text = re.sub(r', [a-zA-Z]* reported today\.$', '.', text)
    text = re.sub(r', reported [a-zA-Z]*\.$', '.', text)
    text = re.sub(r', [a-zA-Z]* reports\.$', '.', text)
    text = re.sub(r', reports [a-zA-Z]*\.$', '.', text)
    text = re.sub(r', according to[\s|a-zA-Z]+\.$', '.', text, flags=re.IGNORECASE)
    # mid-sentence
    text = re.sub(r', (\w*\s+){1,2}said,', '', text)

    if  text != content_obj.realized_text:
        Realization.logger.debug("removing attribution from -- {} -- new sentence: {}".format(content_obj.realized_text, text))

    content_obj.realized_text = text
    return content_obj


def use_extra_quota_space(realized_sents, overflow_sents, starting_sents):
    remaining_quota = WORD_QUOTA - get_num_words_in_collection(realized_sents)
    will_fit = will_fit_sents(overflow_sents, remaining_quota)
    will_fit_non_duplicates = filter_out_duplicates(realized_sents, will_fit)

    if will_fit_non_duplicates:
        return realized_sents + [sorted(will_fit_non_duplicates, key=lambda x: x.score, reverse=True)[0]]
    return realized_sents


def will_fit_sents(sents, remaining_quota):
    return [sent for sent in sents if content_obj_word_count(sent) <= remaining_quota]


def filter_out_duplicates(sent_texts, overflow_options):
    return [overflow_sent for overflow_sent in overflow_options
            if not is_duplicate(overflow_sent, sent_texts)]


def is_duplicate(overflow, sents):
    return any([redundant_lexical_overlap(sent, overflow, 0.84) for sent in sents])


def remove_sentences_starting_with_pronouns_removal_funct(content_obj):
    w1 = content_obj.span[0].text.lower()
    w2 = content_obj.span[1].text.lower()

    #Below, a list of exceptions that should be left in
    if (w1 == "there" and w2 == "are") or \
            (w1 == "there" and w2 == "'re") or \
            (w1 == "there" and w2 == "is") or \
            (w1 == "there" and w2 == "'s") or \
            (w1 == "it" and w2 == "is") or \
            (w1 == "it" and w2 == "'s") or \
            (w1 == "it" and w2 == "has") or \
            (w1 == "it" and w2 == "was") or \
            (w1 == "this" and w2 == "is") or \
            (w1 == "that" and w2 == "'s") or \
            (w1 == "that" and w2 == "is"):
        return False

    #spaCy parsers isn't perfect, so the rules below are based on found parses.
    starting_pos = content_obj.span[0].pos_
    starting_tag = content_obj.span[0].tag_
    starting_dep = content_obj.span[0].dep_
    second_pos = content_obj.span[1].pos_
    second_tag = content_obj.span[1].tag_
    second_dep = content_obj.span[1].dep_
    if starting_pos == "DET" and starting_dep == "nsubj": #Phrases such as "That suddenly created a more serious situation at Mount St. Helens, the most active volcano in the lower 48 states."
        return True
    elif starting_pos == "ADV" and starting_dep == "nsubj":  # Phrases such as "Most had eaten cooked hot dogs the month before they became ill."
        return True
    elif starting_pos == "PRON": #Phrases like "He faces life in prison if he is convicted of murder."
        return True
    elif starting_pos == "ADV" and second_tag == "VBZ": #Phrases like "Here is the progress on a few of the important goals for the bay, based on information from the EPA's Chesapeake Bay Program."
        return True

def remove_sentences_starting_with_pronouns(sents):
    return removal_step(remove_sentences_starting_with_pronouns_removal_funct, sents, 'remove_sentences_starting_with_pronouns')
