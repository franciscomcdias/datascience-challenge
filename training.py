"""
TRAINING.PY

usage: `python training.py [DATAFRAME FILE] --type=(tfidf|convnet) --model=[MODEL FILE]`

Trains a predictive model for reviews text categorization.
The model uses a supervised method for learning the most probable categories.
This method requires as input a set of texts already categorized.

This script trains 2 types of models:
- tf-idf + svm
- CNN

author: mail@franciscodias.pt
date: 03-04-2017
version: 1.0
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd
from gensim.models.keyedvectors import Word2VecKeyedVectors
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, Dense, Convolution1D, MaxPooling1D, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import common

#convnet parameters
DROPOUT_VAL = 0.1
CLASSES_DIM = 6
SEQUENCE_DIM = 200

# paths to the word2vec model
KEY_VECS = "models/wv/word.vectors"
WEIGTHS = "models/wv/word.vectors.vectors.npy"

# logger
logger = logging.getLogger('bunch')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))


# TFIDF RELATED ##

def build_tfidf_model():
    """

    :return:
    """
    # TODO: arguments to change the parameters of the model
    count_vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=500)
    tf_idf = TfidfTransformer(norm="l2")
    svm_clf = SVC()
    model = Pipeline([
        ('count_vectorizer', count_vectorizer),
        ('tf_idf', tf_idf),
        ('svm_clf', svm_clf)
    ])
    return model


def train_tfidf_model(model, train_data, train_labels):
    """

    :param model:
    :param train_data:
    :param train_labels:
    :return:
    """
    model.fit(train_data, train_labels)
    return model


# CONVNET RELATED ##

def build_convnet_model(weights, sequence_dim, classes_dim, dropout_val):
    """

    :param weights:
    :param sequence_dim:
    :param classes_dim:
    :param dropout_val:
    :return:
    """
    _lexicon_dim = weights.shape[0]
    _embed_dim = weights.shape[1]
    inputs = Input(shape=(sequence_dim,), dtype='int32')
    embed = Embedding(input_dim=_lexicon_dim, output_dim=_embed_dim, weights=[weights])(inputs)

    x = Convolution1D(100, 5, activation='relu')(embed)
    x = MaxPooling1D(5)(x)
    x = Convolution1D(100, 5, activation='relu')(x)
    x = MaxPooling1D(4)(x)
    x = Convolution1D(100, 5, activation='relu')(x)
    x = MaxPooling1D(3)(x)

    flatten = Flatten()(x)
    dropout = Dropout(dropout_val)(flatten)
    x = Dense(100, activation='relu')(dropout)

    outputs = Dense(classes_dim, activation='softmax', name="last_layer")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-08), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_word2vec(key_vecs_file, weights_file):
    """

    :param key_vecs_file:
    :param weights_file:
    :return:
    """
    logger.info("loading word2vec model...")
    wv = Word2VecKeyedVectors.load(key_vecs_file)
    weights = np.load(weights_file)
    return wv, weights


def create_embedding_vectors(df):
    """

    :param df:
    :return:
    """

    def vectorise_token(token, wv):
        # just ignoring OOVs
        vector = None
        if token in wv.vocab:
            vector = wv.index2entity.index(token)
        return vector

    logger.info("creating embedding vectors...")
    embedding_vectors = []
    for review in df:
        review_embedding = []
        for token in review:
            vector = vectorise_token(token, wv)
            if vector is not None:
                review_embedding.append(vector)
        embedding_vectors.append(review_embedding)
    return embedding_vectors


def train_convnet_model(model, train_vectors, train_labels):
    """

    :param model:
    :param train_vectors:
    :param train_labels:
    :return:
    """
    early_stop_cb = EarlyStopping(patience=7, monitor='val_acc', mode='max')
    callbacks = [early_stop_cb]
    history = model.fit(
        train_vectors, train_labels,
        callbacks=callbacks,
        epochs=100, validation_split=0.10,
        shuffle=True, batch_size=50
    )
    return model, history


def load_dataframe(dataframe_file):
    """

    :param dataframe_file:
    :return:
    """
    logger.info("loading dataframe...")
    return pd.read_pickle(dataframe_file)


def preprocess_dataframe(df):
    """
    Pre-processing of the dataframe data:
    - remove entries with no category
    - remove duplicate entries
    - remove entries with possible non-english texts
    :param df: pandas df
    :return: pandas df
    """
    # remove entries with no category
    df = df[df.labelmax != "null"]
    # remove duplicates
    df = df.drop_duplicates()
    # remove entries with possible non-english texts
    df = df[df["text"].apply(common.text_language) == "en"]
    return df


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataframe', help='model to load and perform', type=str)
    parser.add_argument('type', type=str, help='type of model, "tfidf" or "convnet"')
    parser.add_argument(''
                        'model', type=str, help='file path for the serialized model')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # arguments ---
    dataframe_file = args.dataframe
    model_file = args.model
    model_type = args.type

    if model_type not in ["convnet", "tfidf"]:
        raise Exception("unknown type")

    # dataframe ---
    df = load_dataframe(dataframe_file)
    df = preprocess_dataframe(df)

    if model_type == "convnet":
        wv, weights = load_word2vec(KEY_VECS, WEIGTHS)
        build_convnet_model(weights, SEQUENCE_DIM, CLASSES_DIM, DROPOUT_VAL)

    elif model_type == "tfidf":
        pass
