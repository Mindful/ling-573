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

        if Realization.config['dump_selected_content']:
            self.dump_selected_content()

        if Realization.config['log_realization_changes']:
            self.log_realization_changes()


    def log_realization_changes(self):
        for content_object in self.realized_content:
            Realization.logger.info("Article ID: {}".format(content_object.article.id))
            Realization.logger.info("\tInput Text: {}".format(content_object.span.text))
            Realization.logger.info("\tOutput Text: {}".format(content_object.realized_text))

        initial_word_count = get_num_words_in_collection(self.selected_content.selected_content)
        Realization.logger.info("\tNarrowed from {} words to {} words".format(initial_word_count, self.output_words))
        Realization.logger.info("\tNarrowed from {} sentences to {} sentences".format(len(self.selected_content.selected_content), len(self.realized_content)))


    def dump_selected_content(self):
        for content_object in self.selected_content.selected_content:
            Realization.logger.info("article id: " + str(content_object.article.id))
            Realization.logger.info("span: " + str(content_object.span))


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
        if Realization.config['remove_subjectless_sentences']:
            cand_sents = remove_subjectless_sentences(cand_sents)
        if len(Realization.config['remove_full_spans_that_match']) > 0:
            cand_sents = filter_content_by_regex_list(cand_sents, Realization.config['remove_full_spans_that_match'])

        cand_sents = remove_stranded_colon_sents(cand_sents)
        cand_sents = remove_fragments(cand_sents)

        ## trim_content_objs modifies the realized_text on each content object. span no longer reliable
        cand_sents = trim_content_objs(cand_sents)

        if Realization.config['remove_attributions']:
            cand_sents = remove_attributions(cand_sents)

        if len(Realization.config['remove_subspans_that_match']) > 0:
            cand_sents = remove_text_by_regex_list(cand_sents, Realization.config['remove_subspans_that_match'])

        unique_sents, overflow_sents = remove_redundant_sents(cand_sents)

        removed = []
        total_words = 0

        for content in unique_sents:
            remaining_words = WORD_QUOTA - total_words
            text_len = len(content.realized_text.split())
            if text_len <= remaining_words and text_len > Realization.config['minimum_sentence_length']:
                total_words += text_len
            else:
                removed.append(content)

        self.output_words = total_words
        return_content = [content for content in unique_sents if content not in removed]
        return_content = clean_up_objects(return_content)
        return_content = use_extra_quota_space(return_content, overflow_sents, selection_object.selected_content)
        return return_content


def use_extra_quota_space(realized_sents, overflow_sents, starting_sents):
    sent_texts = [sent.realized_text for sent in realized_sents]
    remaining_quota = WORD_QUOTA - get_num_words_in_collection(realized_sents)
    will_fit = will_fit_sents(overflow_sents, remaining_quota)
    will_fit_non_duplicates = filter_out_duplicates(sent_texts, will_fit)

    if will_fit_non_duplicates:
        return realized_sents + [sorted(will_fit_non_duplicates, key=lambda x: x.score, reverse=True)[0]]
    return realized_sents


def will_fit_sents(sents, remaining_quota):
    return [sent for sent in sents
            if len(sent.realized_text.split()) <= remaining_quota]


def filter_out_duplicates(sent_texts, overflow_options):
    return [sent for sent in overflow_options if sent.realized_text not in sent_texts]


def remove_questions(content_objs):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, 'remove_questions'):
        return content_objs

    words_trimmed = 0
    removed = []

    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            break
        if re.match(".*\?.*",content_obj.span.text) is not None:
            removed.append(content_obj)
            if Realization.config['log_realization_changes']:
                Realization.logger.info("Removing sentence with question: {}".format(content_obj.span.text))

    return [content for content in content_objs if content not in removed]


def no_extra_remaining(trimming_quota, category):
    if trimming_quota <= 0:
        if Realization.config['log_realization_changes']:
            Realization.logger.info('{}: already hit trimming limit. skip method.'.format(category))
        return True
    return False


def remove_quotes(content_objs):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, 'remove_quotes'):
        return content_objs

    words_trimmed = 0
    removed = []
    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            if Realization.config['log_realization_changes']:
                Realization.logger.info("remove_quotes: hit trimming limit. skip rest of method.")
            break
        if content_obj.span._.contains_quote:
            removed.append(content_obj)
            if Realization.config['log_realization_changes']:
                Realization.logger.info("Removing sentence with quotation: {}".format(content_obj.span.text))

    return [content for content in content_objs if content not in removed]


def remove_stranded_colon_sents(content_objs):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, 'remove_quotes'):
        return content_objs

    words_trimmed = 0
    removed = []
    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            break
        if content_obj.span.text[-1] == ':':
            removed.append(content_obj)
    return [content for content in content_objs if content not in removed]


def remove_fragments(content_objs):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, 'remove_quotes'):
        return content_objs

    words_trimmed = 0
    removed = []
    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            break
        if content_obj.span.text[-1] != '.' and content_obj.span.text[-1] != ';':
            removed.append(content_obj)
    return [content for content in content_objs if content not in removed]


def trim_content_objs(content_objs):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, 'trim_content_objs'):
        return content_objs
    words_trimmed = 0
    new_content_objs = []
    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            if Realization.config['log_realization_changes']:
                Realization.logger.info("new_content_objs: hit trimming limit. skip rest of method.")
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
    for i, content_obj in enumerate(content_objs):
        new_text = clean_up_sentence(content_obj.realized_text)
        content_obj.realized_text = new_text
        new_content_objs.append(content_obj)
    return new_content_objs


def clean_up_sentence(sentence):
    '''
    ** Take care of little details to make sentence readable and grammatical after editing **
    :param sentence: string, the sentence for the summary
    :return: string in printable form
    '''
    ret = set_initial_word_to_upper(sentence)
    return remove_extra_spaces(ret)

def remove_extra_spaces(sentence):
    '''
    ** remove extra spaces from sentence, namely final spaces before punctuation **
    :param sentence: string
    :return: string
    '''
    ret = re.sub("  +"," ", sentence)
    return re.sub(r" +([.?!,])","\g<1>", ret)


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
        exceptions = ('Now')

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


def remove_redundant_sents(content_objs,max_length = 100):
    # will want to factor in weights to determine which sentence to remove, once weights are available
    similarity_metric = Realization.config['similarity_metric']
    similarity_threshold = Realization.config['similarity_threshold']

    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - max_length
    if no_extra_remaining(stop_after_trimming, 'remove_redundant_sents'):
        return content_objs

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
                    if is_redundant(content_obj.span, compare_obj.span,
                                similarity_metric=similarity_metric,
                                similarity_threshold=similarity_threshold) \
                                and compare_obj not in removed:
                        removed.append(compare_obj)
                        words_trimmed += len(compare_obj.span.text.split())

                        if Realization.config['log_realization_changes']:
                            Realization.logger.info("Removing redundant sentence")
                            Realization.logger.info("Kept sentence: " + content_obj.span.text)
                            Realization.logger.info("Removed sentence: " + compare_obj.span.text)

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
    sent_1_tokens = [tok.text.lower() for tok in sent_1]
    sent_2_tokens = [tok.text.lower() for tok in sent_2]
    token_overlap = list(set(sent_1_tokens) & set(sent_2_tokens))

    if len(token_overlap) / len(sent_2_tokens) > 0.72:
        return True

    if similarity_metric=='spacy':
        sim = spacy_similarity(sent_1, sent_2)
    elif similarity_metric=='bert':
        sim = bert_similarity(sent_1, sent_2)
    else:
        #invalid metric string. Give warning?
        sim = -1
    return sim > similarity_threshold

def spacy_similarity(sent_1, sent_2):
    if sent_1.has_vector and sent_2.has_vector:
        # current value (0.87) is chosen by manual inspection of ~20 sentence pairs
        # stripping down to lemmas and removing stop words did NOT seem to help i.e. nlp(" ".join([tok.lemma_ for tok in sent_1 if tok.text not in spacy_stopwords and not tok.is_punct]))
        # might consider adding comparison of doc.ents or doc.noun_chunk overlap
        # this threshold value likely needs to be tuned based on specific content selection strategy
        return sent_1.similarity(sent_2)
    return -1


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


def filter_content_by_regex_list(content_objs,regex_list,max_length = 100):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - max_length

    if no_extra_remaining(stop_after_trimming, 'filter_content_by_regex_list'):
        return content_objs

    words_trimmed = 0
    removed = []
    for i, content_obj in enumerate(content_objs):
        if words_trimmed >= stop_after_trimming:
            if Realization.config['log_realization_changes']:
                Realization.logger.info("filter_content_by_regex_list: hit trimming limit. skip rest of method.")
            break
        for regex in regex_list:
            if re.match(regex,content_obj.span.text) is not None:
                removed.append(content_obj)
                words_trimmed += len(compare_obj.span.text.split())
                if Realization.config['log_realization_changes']:
                    Realization.logger.info("Removing sentence with matching regex: "+
                                            content_obj.span.text)
                break
    return [content for content in content_objs if content not in removed]


def remove_text_by_regex_list(content_objs,regex_list,max_length = 100):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - max_length

    if no_extra_remaining(stop_after_trimming, 'remove_text_by_regex_list'):
        return content_objs

    words_trimmed = 0
    new_content_objs = []
    for i, content_obj in enumerate(content_objs):
        if words_trimmed >= stop_after_trimming:
            if Realization.config['log_realization_changes']:
                Realization.logger.info("remove_text_by_regex_list: hit trimming limit. skip rest of method.")
            new_content_objs.append(content_objs)
            continue
        for regex in regex_list:
            if re.search(regex,content_obj.realized_text) is not None:
                new_text = re.sub(regex,"",content_obj.realized_text)
                if Realization.config['log_realization_changes']:
                    Realization.logger.info("Changing text: -- " +
                                            content_obj.span.text+" -- to "+
                                            new_text)
                content_obj.realized_text = new_text
        new_content_objs.append(content_obj)
    if words_trimmed >= stop_after_trimming:
        if Realization.config['log_realization_changes']:
            Realization.logger.info("remove_text_by_regex_list: hit trimming limit. skip rest of method.")
    return new_content_objs


def get_num_words_in_collection(content_objs):
    word_counts = [len(content.realized_text.split()) for content in content_objs]
    return sum(word_counts)


def remove_subjectless_sentences(content_objs,max_length=100):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - max_length
    if stop_after_trimming <= 0:
        if Realization.config['log_realization_changes']:
            Realization.logger.info("remove_subjectless_sentences: already below word limit. skip method.")
        return content_objs
    words_trimmed = 0
    removed = []
    for i, content_obj in enumerate(content_objs):
        if words_trimmed >= stop_after_trimming:
            if Realization.config['log_realization_changes']:
                Realization.logger.info("remove_subjectless_sentences: hit trimming limit. skip rest of method.")
            break
        found_subj = False
        for tok in content_obj.span:
            if tok.dep_ in ('nsubj','nsubjpass'):
                found_subj = True
                break
        if found_subj == False:
            removed.append(content_obj)
            if Realization.config['log_realization_changes']:
                Realization.logger.info("Removing sentence with no subject: "+
                                        content_obj.span.text)
    return [content for content in content_objs if content not in removed]


def remove_attributions(content_objs):
    current_len = get_num_words_in_collection(content_objs)
    stop_after_trimming = current_len - WORD_QUOTA

    if no_extra_remaining(stop_after_trimming, 'remove_text_by_regex_list'):
        return content_objs

    words_trimmed = 0
    new_content_objs = []

    for content_obj in content_objs:
        if words_trimmed >= stop_after_trimming:
            new_content_objs.append(content_objs)
            continue

        new_text = content_obj.realized_text
        # sentence final
        new_text = re.sub(r', citing [\s|a-zA-Z]+([!.])$', r'\1', new_text, flags=re.IGNORECASE)
        new_text = re.sub(r', [a-zA-Z]* said([!.])$', r'\1', new_text)
        new_text = re.sub(r', [a-zA-Z]*\s([a-zA-Z]*\s)?([a-zA-Z]*\s)?said[!.]$', r'\1', new_text)
        new_text = re.sub(r', [a-zA-Z]* said today([!.])$', r'\1', new_text)
        new_text = re.sub(r', said [a-zA-Z]*([!.])$', r'\1', new_text)
        new_text = re.sub(r', [a-zA-Z]* press reported today([!.])$', r'\1', new_text)
        new_text = re.sub(r', [a-zA-Z]* reported([!.])$', r'\1', new_text)
        new_text = re.sub(r', [a-zA-Z]* reported today([!.])$', r'\1', new_text)
        new_text = re.sub(r', reported [a-zA-Z]*([!.])$', r'\1', new_text)
        new_text = re.sub(r', [a-zA-Z]* reports([!.])$', r'\1', new_text)
        new_text = re.sub(r', reports [a-zA-Z]*([!.])$', r'\1', new_text)
        new_text = re.sub(r', according to[\s|a-zA-Z]+([!.])$', r'\1', new_text, flags=re.IGNORECASE)

        # mid-sentence
        new_text = re.sub(r', (\w*\s+){1,2}said,', '', new_text)

        if Realization.config['log_realization_changes'] and new_text != content_obj.realized_text:
            Realization.logger.info("removing attribution from -- {} -- new sentence: {}".format(content_obj.realized_text, new_text))

        content_obj.realized_text = new_text
        new_content_objs.append(content_obj)

    return new_content_objs

