"""
COMMON.PY

Set of functions shared by the predictor and trainer scripts.

author: mail@franciscodias.pt
date: 03-04-2017
version: 1.0
"""
import logging
import re
import sys

import langdetect
import nltk
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from src import constants

#
stop_words_en = set(stopwords.words("english"))
stemmer_en = SnowballStemmer('english')

# logger
logger = logging.getLogger('bunch')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))


def text_language(text):
    """
    returns the ISO code for the most probable language of the text
    :param text: str
    :return: str (like "en")
    """
    return langdetect.detect(text)


def filter_by_segment(entry, index):
    """
     returns a segment by it's index in a string splited by SEGMENT_TITLES (i.e.: pros= 2, cons = 4, ...)
    :param entry:
    :param index:
    :return:
    """
    segments = re.split(constants.SEGMENT_TITLES, entry)
    
    return segments[index]


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


def tokenisation_posfilter_stemming(text):
    """
    performs tokenisation + gather just words with POS NN/JJ/VB* + stemming
    :param text: str
    :return: str
    """
    # tokenize and remove empty entries
    tokens = nltk.word_tokenize(text.lower())

    # choosing only nouns, verbs, and adjectives
    tokens = [_pos[0] for _pos in pos_tag(tokens) if _pos[1] in constants.ALLOWED_POS]

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


def tokenise_pos_stemming_pros_cons(df):
    """
    performs tokenisation + gather POS NN/JJ/VB* + stemming just for Pros and Cons texts and
    returns the same dataframe with a new column "all_tok_pos_stem" with the concatenation of
    this information
    :param df: pandas df
    :return: pandas df
    """
    logger.info("tok+pos+stem...")

    # pros
    df["pros_tok_pos_stem"] = df["pros"].apply(tokenisation_posfilter_stemming)
    df["pros_tok_pos_stem"] = df["pros_tok_pos_stem"].apply(lambda entry: prefix(entry, "+"))

    # cons
    df["cons_tok_pos_stem"] = df["cons"].apply(tokenisation_posfilter_stemming)
    df["cons_tok_pos_stem"] = df["cons_tok_pos_stem"].apply(lambda entry: prefix(entry, "-"))

    # pros + cons
    df["all_tok_pos_stem"] = df["pros_tok_pos_stem"] + " " + df["cons_tok_pos_stem"]

    return df


def load_word_vectors(key_vecs_file, weights_file):
    """
    loads w2v keyvecs and lexicon into memory
    :param key_vecs_file: path to keyvecs file
    :param weights_file: path to lexicon w2v file
    :return: keyvecs, lexicon
    """
    logger.info("loading word2vec model...")

    wv = Word2VecKeyedVectors.load(key_vecs_file)
    weights = np.load(weights_file)

    return wv, weights


def create_embedding_vectors(df, wv):
    """
    create vectors of words indexes in a w2c lexicon
    Given a dataframe wirh texts
    :param df: pandas df
    :return: lists of int
    """

    def get_w2v_token_index(token, wv):
        """
        Gets the index from a w2v lexicon, if exists
        :param token: str
        :param wv: w2v lexicon list
        :return: int
        """
        # just ignoring OOVs
        index = None
        if token in wv.vocab:
            index = wv.index2entity.index(token)
        return index

    logger.info("creating embedding vectors...")
    embedding_vectors = []

    for text in df:
        text_embedding = []

        for token in text:
            index = get_w2v_token_index(token, wv)
            if index is not None:
                text_embedding.append(index)

        embedding_vectors.append(text_embedding)

    return embedding_vectors
