import re

def is_countworthy_token(token):
    if token.is_punct or token.is_stop or token.like_num or token.like_url or token.like_email:
        return False

    # do exact checks first because 'x in y' python built-in is ~4 times faster than regex search
    exact_checks = ['--', '~', '$', '..', '(', ')', '|', '%', '\\', '/', '^', '@', '#', '!', '+', ',', ';', ':', '{', '}']
 
    for pattern in exact_checks:
        if pattern in token.text:
            return False

    regexes = [r'[A-Z|a-z]+[0-9]{2,}.*', r'.*-', r'\s+', r'^-.*', r'(\w)\1{2,}', r'.*\..*\.', r'^\d*\.', r'^\d']

    for pattern in regexes:
        if re.search(pattern, token.text):
            return False
    return True


def clean_text(text, remove_quotes=False):
    # * remove spurious line breaks and tabs
    text = re.sub('\s+\\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub('(\S+)\\n', r'\1 ', text)
    text = re.sub('\s+', ' ', text)

    # taglines, websites, etc.
    text = re.sub(r'.*top news stories.*$', '', text, flags=re.IGNORECASE)  # remove Here are today's top news stories
    text = re.sub(r'^\s*on the net.*$', '', text, flags=re.IGNORECASE) # remove 'on the net' and everything following
    text = re.sub(r'.{0,50}e-?mail address is.{0,100}', '', text, flags=re.IGNORECASE)  # remove email line e.g. Bob Keefe's e-mail address is bkeefecoxnews.com
    text = re.sub(r'^\s*e-?mail.{0,100}', '', text, flags=re.IGNORECASE)  # remove email and following up to 100 chars, e.g. E-mail: triggp(at)nytimes.com.
    text = re.sub(r'^\s*\(e-?mail.{0,100}\)', '', text, flags=re.IGNORECASE)  # remove email and following up to 100 chars, e.g. (E-mail: triggp(at)nytimes.com.)
    text = re.sub(r'\.\s*e-?mail.{0,100}', '.', text, flags=re.IGNORECASE)  # remove email and following up to 100 chars, e.g. E-mail: triggp(at)nytimes.com. Leave preceeding period.
    text = re.sub(r'^\s*story filed by.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. Story Filed By Cox Newspapers
    text = re.sub(r'^\s*for use by.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. For Use By Clients of the New York Times News Service
    text = re.sub(r'^\s*photos and graphics.{0,100}', '', text, flags=re.IGNORECASE)  # removes e.g. PHOTOS AND GRAPHICS:
    text = re.sub(r'\s*phone:.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. Phone: (888) 603-1036
    text = re.sub(r'\s*pager:.{0,100}', '', text, flags=re.IGNORECASE)  # remove Pager: (800) 946-4645 (PIN 599-4539).
    text = re.sub(r'^\s*technical problems.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. TECHNICAL PROBLEMS:   Peter Trigg
    text = re.sub(r'^\s*questions or.{0,100}', '', text, flags=re.IGNORECASE)  # remove e.g. QUESTIONS OR RERUNS:
    text = re.sub(r"^\s*With photo\s*(\w+\s*){0,5}\.?", "", text, flags=re.IGNORECASE) # remove With photo 
    text = re.sub(r'^\s*With photo.', '', text, flags=re.IGNORECASE)  # remove With photo.
    text = re.sub(r'^https?://\S*$', '', text) # remove urls
    text = re.sub(r'www.{4,100}$', '', text) # remove urls II
    text = re.sub(r'https?://www.{4,100}$', '', text) # remove urls III, e.g. http://www.fs.fed.us/gpnf/mshnvm/
    text = re.sub(r'^\s*[A-Z|\-|\s]{2,50} \(Undated\)\s*', '', text) # remove e.g. BKN-PREVIEW-WEST (Undated)
    text = re.sub(r"\s+\.\s+\.\s+\.", ".", text) # remove . . . 
    text = re.sub(r"\(\s*\)", "", text) # remove ( )
    text = re.sub(r"^\s*(SOURCES:)+\s*(\w+\s*){0,6},?\sstaff reporting.?", "", text, flags=re.IGNORECASE) # remove staff reporting, e.g. Epa chesapeake bay program, staff reporting
    text = re.sub(r",?\sstaff reporting.?", "", text, flags=re.IGNORECASE) # remove staff reporting not at beginning of sentence
    text = re.sub(r'^By [\w+] ([\w+])? ([\w+])?.$', '', text, flags=re.IGNORECASE)  # remove by lines, e.g. By Sam Howe Verhovek.


    # remove (or convert?) the location parenthetical that begins most articles, e.g. "LITTLETON, Colo. (AP) --"
    text = re.sub(r'^.{0,50}\(AP\) (--)*', '', text) # remove (AP) -- and previous text for anything up to 50 chars from beginning of line, 
    text = re.sub(r'^[A-Z]{2,}[,|\w|\d|\s]*\(\w+\)\s*--', '', text)  # remove loc e.g. BANGKOK, April 2 (Xinhua) --
    text = re.sub(r'^[A-Z|\-|\.|,|\s]{2,50}\s+[A-Z|a-z|\.]{0,50}\s+--', '', text) # remove loc and --  e.g. WEST PALM BEACH, Fla. -- 
    text = re.sub(r'^.{0,50}\(JP\):? (--)*', '', text) # remove (JP) -- and previous text for anything up to 50 chars from beginning of line, e.g JAKARTA (JP)
    text = re.sub(r'^[A-Z|\-|\.|\s]{2,100}\(.+\)\s*_', '', text)  # remove loc e.g.   FED-GREENSPAN (Undated) _ 
    text = re.sub(r'^\s*_+', '', text) # remove starting underscore, e.g. "_ The protocol obliges industrialized "
    text = re.sub(r'^[A-Z]{2,}[A-Z |\w|,|\.]*_', '', text) # remove location and underscore, e.g. "NEW YORK _", but don't remove "The letter _ seen by The Associated Press _ said senior leaders"
    text = re.sub(r'^[A-Z]{0,50} --', '', text) # remove loc, e.g. ATLANTA --
    text = re.sub(r'^\([A-Za-z]{0,50}\)--', '', text) # remove loc, e.g. (tampa)--
    text = re.sub(r'.{0,10}\(JP\):\s?', '', text) # remove Jakarta (JP): 

    # total junk, no idea
    text = re.sub(r'^\s*\(?[A-Za-z]{2}\/[A-Za-z0-9]{2,4}\)?$', '', text) # remove e.g. po/pi04, em/ea04, (lc/ml)
    text = re.sub(r'^\s*\(?[a-z]{2}\-[a-z0-9]{2,3}\s?(\/[a-z]{2,3})?\)?$', '', text) # remove e.g. sn-sjs, (pd-fg/imj), (mb-mn/pp)
    text = re.sub(r'^\s*[a-z]{2}[0-9]{2}$', '', text) # remove e.g. js04
    text = re.sub(r'^\s*nn\s*$', '', text)  # remove nn lines
    text = re.sub(r'^\s*(\s*-\s*)+\s*$', '', text)  # remove  - - - - lines
    text = re.sub(r'^\.+$', '', text)  # remove empty ... lines, e.g. "." or "..."
    text = re.sub(r'[A-Z|a-z]*-[A-Z|a-z]*-[A-Z|a-z]*-[A-Z|a-z]*\s+', '', text)  # remove Bc-fla-lafave-deal 
    text = re.sub(r'\( \) --', '', text)  # remove ( ) --
    text = re.sub(r'\s+_\s+', '', text)  # remove stranded underscores


    # news things
    text = re.sub(r'^\s*with\s*[\w | -]{0,50}.?$', '', text, flags=re.IGNORECASE) # e.g. With a map-graphic., With, With map.
    text = re.sub(r'^[\w|\s]*NewsBrief by.{0,100}$', '', text, flags=re.IGNORECASE) # e.g. AP NewsBrief by GABRIEL MADWAY
    text = re.sub(r'^ENDIT$', '', text)  # remove ENDIT
    text = re.sub(r'^\(Begin optional trim\)$', '', text)  # remove (Begin optional trim)
    text = re.sub(r'^\(Optional add end\)$', '', text)  # remove (Optional add end)
    text = re.sub(r'^\(STORY CAN END HERE. OPTIONAL MATERIAL FOLLOWS\)$', '', text)  # remove (Optional add end)

    # senticizing issue stop-gaps
    text = re.sub(r'[\w]* Not Otherwise Specified', 'not otherwise specified', text)  # remove Here are today's top news stories
    text = re.sub(r'([\w]*)\s?\([A-Z]{2,4}\)\s?', r",\1 ", text)  # remove all caps parenthetical abbreviations


    # standarize quotation marks, i.e. `` -> "   and '' -> "
    text = re.sub("([^`])`([^`])", r"\1'\2", text) # deal with opening nested quotes, e.g. ``Having exercised that right, they cannot then say, `We're police officers, therefore what we did was OK,''' he said.
    text = re.sub("'''", "'\"", text) # deal with closing nested quotes
    text = re.sub("``", '"', text)  # ``It's like a... 
    text = re.sub("''", '"', text)  #  ...like a prison in there,'' said Jessica Miller, 15.
    text = re.sub(r',\s([\"|\'])\s', r",\1 ", text)
    text = re.sub(r'\!\"( \w+)', r'!", \1 ', text)

    if remove_quotes:
        text = re.sub("\"", '', text)
        text = re.sub(r"\s'([A-Z|a-z])", r" \1", text)
        text = re.sub(r"([A-Z|a-z])'\s", r"\1 ", text)


    # * remove spurious line breaks and tabs
    text = re.sub('\s+\\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub('(\S+)\\n', r'\1 ', text)
    text = re.sub('\s+', ' ', text)

    return text.strip()


# potentially add for (?):
# If you have questions, please call 202-334-7666 or 213-237-7832 or e-mail latwp(at)washpost.com or latwp(at)latimes.com.\# SPORTS ("s" category)
# BKN-ROCKETS-BULLS (Chicago) -- 
# WASHINGTON ("w" category)
# (Jim Armstrong is a columnist for The Denver Post. Contact him at jarmstrongdenverpost.com.)
