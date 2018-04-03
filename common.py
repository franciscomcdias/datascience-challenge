"""
COMMON.PY

Set of functions used by the predictor and trainer.

author: mail@franciscodias.pt
date: 03-04-2017
version: 1.0
"""
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import langdetect

stop_words_en = set(stopwords.words("english"))
stemmer_en = SnowballStemmer('english')

CLASS_LABELS = ['adaptability', 'collaboration', 'customer', 'detail', 'integrity', 'result']


def text_language(text):
    """
    returns the ISO code for the most probable language of the text
    :param text: str
    :return: str (like "en")
    """
    return langdetect.detect(text)


def tok_stem_stop(text):
    """
    performs tokenisation + stemming + stop-word removal
    :param text: str
    :return: str
    """
    # tokenize and remove empty entries
    tokens = nltk.word_tokenize(text.lower())
    # stop-word removal, single character tokens removal, and stemming
    tokens = [stemmer_en.stem(_token) for _token in tokens
              if _token not in stop_words_en
              and len(_token) > 1
              and _token not in ["'s", "'d"]
              ]
    # n't -> not
    tokens = [_token.replace("n't", "not") for _token in tokens]
    return " ".join(tokens)


def tok_pos_stem(text):
    """
    performs tokenisation + gather just words with POS NN/JJ/VB* + stemming
    :param text: str
    :return: str
    """
    # tokenize and remove empty entries
    tokens = nltk.word_tokenize(text.lower())
    # choosing only nouns, verbs, and adjectives
    tokens = [_pos[0] for _pos in pos_tag(tokens) if _pos[1] in ["NN", "NNS", "JJ", "VB", "VBG"]]
    # stemming
    tokens = [stemmer_en.stem(_token) for _token in tokens]
    return " ".join(tokens)


def tokens(text):
    """
    performs tokenisation + lower case
    :param text: str
    :return: list of str
    """
    return nltk.word_tokenize(text.lower())


def prefix(text, _prefix):
    """
    adds a prefix (e.g.: +/-) to a token
    :param text: str
    :param _prefix: str
    :return: str
    """
    return " ".join([_prefix + _token for _token in text.split()])
