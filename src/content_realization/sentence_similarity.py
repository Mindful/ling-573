from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]


def is_redundant(sent_1, sent_2,
                 similarity_metric='spacy',
                 similarity_threshold=.97):
    '''
    ** Given two sentences, check if they're redundant
    :param sent_1: a spacy span built from a sentence
    :param sent_2: a second spacy span built from another sentence
    :param similarity_metric: what similarity measure to use. 'spacy' or 'bert'
    :param similarity_threshold: what % similarity is the cutoff for redundancy
    :return: True/False for redundant/not redundant
    '''
    if similarity_metric=='spacy':
        sim = spacy_similarity(sent_1,sent_2)
    elif similarity_metric=='bert':
        sim = bert_similarity(sent_1, sent_2)
    else:
        #invalid metric string. Give warning?
        sim = -1
    return sim > similarity_threshold

def spacy_similarity(sent_1,sent_2):
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

    #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    #model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

    #classes = ["not paraphrase", "is paraphrase"]
    '''
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
    '''
    sequence_0 = sent_1.text
    sequence_1 = sent_2.text
    possible_paraphrase = tokenizer.encode_plus(sequence_0,sequence_1,return_tensors="pt")

    paraphrase_classification_logits = model(**possible_paraphrase)[0]
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0][1] #[1] is the class for "is paraphrase"
    '''
    paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, return_tensors="pt")
    not_paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, return_tensors="pt")
    paraphrase_classification_logits = model(**paraphrase)[0]
    not_paraphrase_classification_logits = model(**not_paraphrase)[0]

    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

    print("Should be paraphrase")
    for i in range(len(classes)):
        print(f"{classes[i]}: {round(paraphrase_results[i] * 100)}%")

    print("\nShould not be paraphrase")
    for i in range(len(classes)):
        print(f"{classes[i]}: {round(not_paraphrase_results[i] * 100)}%")
    '''
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
