def span_len(span_object):
    '''
    :param span_object: a spacy span object
    :return: The number of words in the span.
    '''
    len = 0
    for tok in span_object:
        if not tok.is_punct:
            len+=1
    return len

def word_span(doc_object,start,end):
    '''
    :param doc_object: a spacy doc object to get a spacy span from
    :param start: the start of the span (measured in tokens)
    :param end: the end of the spacy span (measured in words past start)
    :return:  a spacy span object with number of words = end - start
    '''
    span_end = end
    word_count = 0
    ind = start
    while ind < len(doc_object):
        ind +=1
        span = doc_object[ind-1:ind]
        if not span[0].is_punct:
            word_count +=1
        if word_count == (end-start):
            break
    return doc_object[start:ind]




