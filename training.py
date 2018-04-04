"""
TRAINING.PY

usage: `python training.py [DATAFRAME FILE] (tfidf|convnet) [MODEL FILE]`

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

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, Dense, Convolution1D, MaxPooling1D, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import common
import constants

logger = common.logger


# TFIDF MODEL RELATED ##

def build_tfidf_model():
    """
    Assembles a TF-IDF + SVM pipeline.

    :return: model
    """
    # TODO: arguments to change the parameters of the model
    count_vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=500)
    tf_idf = TfidfTransformer(norm="l2")
    svm_clf = SVC()
    model = Pipeline([
        ("count_vectorizer", count_vectorizer),
        ("tf_idf", tf_idf),
        ("svm_clf", svm_clf)
    ])
    return model


def train_tfidf_model(model, df):
    """
    Trains TF-IDF model given a dataframe

    :param model: sklearn pipeline model
    :param df: pandas df
    :return: model
    """
    df = common.tokenise_pos_stemming(df)

    logger.info("training model...")
    train_data = df.as_matrix(columns=["all_tok_pos_stem"])[:, 0]
    train_labels = df.as_matrix(columns=["labelmax"])[:, 0]

    model.fit(train_data, train_labels)
    return model


# CONVNET MODEL RELATED ##

def build_convnet_model(weights, sequence_dim, classes_dim, dropout_val):
    """
    Assembles and compiles a CNN with 8 layers based on Yoon Kim architecture (2014)

    :param weights: word2vec lexicon
    :param sequence_dim: int, input max dimension
    :param classes_dim: int, number of categories at the end
    :param dropout_val: float, dropout prob value
    :return: convnet model
    """
    # TODO: arguments (instead of hard-coded constants) describing the parameters of the model
    _lexicon_dim = weights.shape[0]
    _embed_dim = weights.shape[1]
    inputs = Input(shape=(sequence_dim,), dtype="int32")
    embed = Embedding(input_dim=_lexicon_dim, output_dim=_embed_dim, weights=[weights])(inputs)

    x = Convolution1D(100, 5, activation="relu")(embed)
    x = MaxPooling1D(5)(x)
    x = Convolution1D(100, 5, activation="relu")(x)
    x = MaxPooling1D(4)(x)
    x = Convolution1D(100, 5, activation="relu")(x)
    x = MaxPooling1D(3)(x)

    flatten = Flatten()(x)
    dropout = Dropout(dropout_val)(flatten)
    x = Dense(100, activation="relu")(dropout)

    outputs = Dense(classes_dim, activation="softmax", name="last_layer")(x)

    model = Model(inputs=inputs, outputs=outputs)
    # TODO: should be a constant
    model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-08), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def train_convnet_model(model, train_vectors, train_labels):
    """

    :param model: CNN
    :param train_vectors: ndarray,
    :param train_labels:
    :return: model: CNN, history: dict, training history
    """
    early_stop_cb = EarlyStopping(patience=7, monitor="val_acc", mode="max")
    callbacks = [early_stop_cb]
    history = model.fit(
        train_vectors, train_labels,
        callbacks=callbacks,
        epochs=100, validation_split=0.10,
        shuffle=True, batch_size=50
    )
    print("\n")
    return model, history


def load_dataframe(dataframe_file):
    """

    :param dataframe_file: str, path pickle file
    :return: pandas df
    """
    logger.info("loading dataframe...")
    return pd.read_pickle(dataframe_file)


def preprocess_dataframe(df):
    """
    Pre-processing of the dataframe data:
    -- remove entries with no category
    -- remove duplicate entries
    -- remove entries with possible non-english texts
    -- segment texts between pros and cons

    :param df: pandas df
    :return: pandas df
    """
    logger.info("pre-processing dataframe...")
    # remove entries with no category
    logger.info("\t- no categories")
    df = df[df.labelmax != "null"]
    # remove duplicates
    logger.info("\t- duplicates")
    df = df.drop_duplicates()
    # remove entries with possible non-english texts
    logger.info("\t- non-english")
    df = df[df["text"].apply(common.text_language) == "en"]
    # split between pros and cons
    logger.info("\t- pros/cons")
    df["pros"] = df["text"].apply(lambda entry: common.filter_by_segment(entry, 2))
    df["cons"] = df["text"].apply(lambda entry: common.filter_by_segment(entry, 4))
    df["all"] = df["pros"] + " " + df["cons"]
    return df


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataframe", help="model to load and perform", type=str)
    parser.add_argument("type", type=str, help="type of model, 'tfidf' or 'convnet'")
    parser.add_argument("model", type=str, help="file path for the serialized model")
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
    df = df.truncate(after=200)
    df = preprocess_dataframe(df)

    if model_type == "convnet":
        # CNN

        # builds w2v lexicon
        wv, weights = common.load_word2vec(constants.KEY_VECS, constants.WEIGTHS)
        embedding_vectors = common.create_embedding_vectors(df["all"], wv)

        # converts training data into input format for CNN
        train_vectors = pad_sequences(embedding_vectors, maxlen=constants.SEQUENCE_DIM, padding='post')
        train_labelmax = [constants.CLASS_LABELS.index(row) for row in df["labelmax"].as_matrix()]
        train_labels = to_categorical(np.asarray(train_labelmax))

        # builds and train model
        model = build_convnet_model(weights, constants.SEQUENCE_DIM, constants.CLASSES_DIM, constants.DROPOUT_VAL)
        model, _ = train_convnet_model(model, train_vectors, train_labels)

        # stores model
        logger.info("saving " + model_file)
        model.save(model_file)

    elif model_type == "tfidf":
        # TF-IDF

        # builds and train model
        model = build_tfidf_model()
        model = train_tfidf_model(model, df)

        # stores model
        logger.info("saving " + model_file)
        joblib.dump(model, model_file)

    logger.info("done!")
