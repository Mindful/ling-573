import lxml.etree as ET
import os.path
from abc import ABC
from data.article import Article


class CorpusFile:
    def __init__(self, location, read_function):
        self.location = location
        self.read_function = read_function

    def get_articles(self):
        return self.read_function(self.location)

    def __eq__(self, other):
        return isinstance(other, CorpusFile) and self.location == other.location

    def __hash__(self):
        return hash(self.location)

    def __repr__(self):
        return "CorpusFile('{}')".format(self.location)


def read_new_content_file(filename):
    root = ET.parse(filename).getroot()

    return [Article.from_new_xml(x) for x in root]

# as per acquaint.dtd
ENTITIES = {
    "AMP": "&amp;",  # to get '&', replace with xml ampersand
    "Cx14": "",
    "Cx13": "",
    "Cx12": "",
    "Cx11": "",
    "Cx1f": "",
    "HT": "",
    "QL": "",
    "QR": "",
    "LR": "",
    "UR": "",
    "QC": ""
}


def read_old_content_file(filename):
    parser = ET.XMLParser(recover=True)
    with open(filename, 'r') as f:
        file_content = f.read()

    for entity, value in ENTITIES.items():
        target = '&'+entity+';'
        file_content = file_content.replace(target, value)

    root = ET.fromstring('<dummy>'+file_content+'</dummy>', parser=parser)

    return [Article.from_old_xml(x) for x in root]


class Corpus(ABC):
    base_directory = '/corpora/LDC/'

    def __init__(self, name, start_year, end_year, directory):
        self.year_range = range(start_year, end_year+1)  # include final year
        self.name = name
        self.directory = directory

    def get_journal_dir(self, query):
        raise RuntimeError("NYI")

    def get_filename(self, query):
        raise RuntimeError("NYI")

    def valid_file(self, filename):
        raise RuntimeError("NYI")

    def get_file_location(self, query):
        file_location = os.path.join(self.get_journal_dir(query), self.get_filename(query))
        return CorpusFile(file_location, self.reader_function)

    def get_all_files(self):
        all_file_locations = []
        for dir, subdirs, subfiles in os.walk(self.directory):
            all_file_locations.extend(os.path.join(dir, f) for f in subfiles if self.valid_file(f))

        return [CorpusFile(file_location, self.reader_function) for file_location in all_file_locations]



class Aquaint(Corpus):
    def __init__(self, base_dir=Corpus.base_directory):
        super().__init__('AQUAINT', 1996, 2000, os.path.join(base_dir, 'LDC02T31/'))
        self.reader_function = read_old_content_file

    def get_journal_dir(self, query):
        return os.path.join(self.directory, query.journal_id.lower())

    def get_filename(self, query):
        parent_dir = str(query.year)
        string_parts = [query.file_id, '_',
                        'XIN' if query.journal_id == 'XIE' else query.journal_id]
        if query.journal_id != 'NYT':
            string_parts.append('_ENG')
        return os.path.join(parent_dir, ''.join(string_parts))

    def valid_file(self, filename):
        return filename[-3:] == 'NYT' or filename[-4:] == '_ENG'


class Aquaint2(Corpus):

    lang_id = 'eng'

    def __init__(self, base_dir=Corpus.base_directory):
        super().__init__('AQUAINT-2', 2004, 2006, os.path.join(base_dir, 'LDC08T25/data/'))
        self.reader_function = read_new_content_file

    def get_journal_dir(self, query):
        return os.path.join(self.directory, query.journal_id.lower() + '_' + self.lang_id)

    def get_filename(self, query):
        return ''.join([query.journal_id.lower(), '_', self.lang_id, '_', query.file_id[:-2], '.xml'])

    def valid_file(self, filename):
        return filename[-4:] == '.xml'

