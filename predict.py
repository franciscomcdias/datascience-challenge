"""
PREDICT.PY

usage: `python predict.py [MODEL FILE] [OUTPUT TSV] [JSON FILES...]`

Runs the predictive model over a set of JSON files containing reviews.

author: mail@franciscodias.pt
date: 03-04-2017
version: 1.0
"""

import argparse
import json

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib

import common
import constants

logger = common.logger


def load_json_review_from_file(file_paths):
    """
    Loads a file with a list of reviews in JSON format.
    Every JSON structure must conatin at least the fields: title, pros, cons
    :param file_paths: list of str
    :return: dataframe containing: file_path, title, pros, and cons
    """
    content = []
    for file_path in file_paths:
        review_company = json.load(open(file_path, "r"))
        for review in review_company:
            title, pros, cons = "", "", ""
            if "title" in review and review["title"]:
                title = review["title"]
            if "pros" in review and review["pros"]:
                pros = review["pros"]
            if "cons" in review and review["cons"]:
                cons = review["cons"]
            content.append({
                "file_path": file_path,
                "title": title,
                "pros": pros,
                "cons": cons,
                "all": pros + " " + cons
            })
    return pd.DataFrame.from_dict(content)


def write_results_into_tsv_file(results, file_path):
    """
    Writes a list of results as a TSV file
    :param results: list of 3-uples of str
    :param file_path: str
    :return: None
    """
    output_tsv = open(file_path, "w")
    for result in results:
        for entry in result:
            output_tsv.write(entry)
            output_tsv.write("\t")
        output_tsv.write("\n")
    output_tsv.close()


def predict_tfidf(model, reviews):
    """
    Runs a prediction of the category of a review using a TFIDF model.
    The categories are listed in CLASS_LABELS in the file `common.py`.
    :param model: serialized model
    :param reviews: dataframe
    :return: list of tuples with: file_path, title of review, predicted category for review
    """
    results = []
    for index, review in reviews.iterrows():
        input = common.tok_pos_stem(review["all"])
        labels = model.predict([input])
        label = np.argmax(labels[0])
        results.append([review["file_path"], review["title"], constants.CLASS_LABELS[label]])
    return results


def predict_keras(model, df, key_vecs_file, weights_file):
    """

    :param model:
    :param reviews: dataframe
    :param key_vecs_file:
    :param weights_file:
    :return:
    """
    results = []
    wv, weights = common.load_word2vec(key_vecs_file, weights_file)
    embedding_vectors = common.create_embedding_vectors(df["all"], wv)

    vectors = pad_sequences(embedding_vectors, maxlen=constants.SEQUENCE_DIM, padding='post')

    logger.info("predicting...")
    for index, review in reviews.iterrows():
        labels = model.predict(vectors)
        label = np.argmax(labels[0])
        results.append([review["file_path"], review["title"], constants.CLASS_LABELS[label]])
    return results


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', help='model to load and perform', type=str)
    parser.add_argument('outtsv', help='output TSV file', type=str)
    parser.add_argument('file', type=str, nargs="+", help='JSON file(s) to process')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    files_to_process = args.file
    model_file = args.model
    outtsv_file = args.outtsv

    reviews = load_json_review_from_file(files_to_process)

    if model_file.endswith(".h5"):
        model = load_model(model_file)
        results = predict_keras(model, reviews, constants.KEY_VECS, constants.WEIGTHS)
    elif model_file.endswith(".pickle"):
        model = joblib.load(model_file)
        results = predict_tfidf(model, reviews)
    else:
        raise Exception("unknown model type")

    logger.info("writing " + outtsv_file)
    write_results_into_tsv_file(results, outtsv_file)

    logger.info("done!")