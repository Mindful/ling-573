import os
import logging
import spacy

from data.corpora import Aquaint, Aquaint2

import yaml

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SOURCE_DIR)

CONFIG_FILENAME = 'config.yaml'
CONFIG_FILE = os.path.join(ROOT_DIR, CONFIG_FILENAME)

LOGGING_FILE = 'summarize.log'
NLP = None

DEV_TEST = 'dev_test'
TRAIN = 'train'

class Globals:
    corpora = [Aquaint(), Aquaint2()]
    datasets = {
        DEV_TEST: '/dropbox/19-20/573/Data/Documents/devtest/GuidedSumm10_test_topics.xml',
        TRAIN: '/dropbox/19-20/573/Data/Documents/training/2009/UpdateSumm09_test_topics.xml'
    }

    nlp = None
    config = None
    logger = None
    idf = None


class PipelineComponent:
    logger = None

    @staticmethod
    def setup():
        pass


def setup(pipeline_classes):
    logging.basicConfig(format='[%(asctime)s - %(name)s::%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    base_logger = logging.getLogger()
    base_logger.setLevel(logging.INFO)
    logger = logging.getLogger('Global')

    with open(CONFIG_FILE, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info("Loading spaCy, this may take a moment...")
    Globals.nlp = spacy.load("en_core_web_lg")
    Globals.config = config[Globals.__name__]
    Globals.logger = logger
    logger.info('Loading idf data...')


    from metric_computation import get_idf  # common is imported in too many places - avoid circular imports
    Globals.idf = get_idf(next(c for c in Globals.corpora if c.name == Globals.config['idf_corpus']),
                          lemmatized=Globals.config['lemmatized_idf'])

    for clazz in pipeline_classes:
        class_name = clazz.__name__
        logger.info('Setting up '+class_name)
        clazz.logger = logging.getLogger(class_name)
        clazz.config = config[class_name]
        clazz.setup()
