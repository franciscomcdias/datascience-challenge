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
import os
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib

import common
import constants

logger = common.logger


def review_from_file(file_path, content):
    """
    Retrieves review texts from a JSON
    :param file_path: str, path to a file
    :param content:
    :return:
    """
    logger.info("processing file " + file_path)
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


def load_json_review_from_file(paths):
    """
    Loads a folder of files or a list of files with reviews in JSON format.
    Every JSON structure must conatin at least the fields: title, pros, cons
    :param paths: list of str, can be a folder or a list of files
    :return: dataframe containing: file_path, title, pros, and cons
    """
    content = []
    for path in paths:
        if os.path.isdir(path):
            for file in os.listdir(path):
                review_from_file(os.path.join(path, file), content)
        else:
            review_from_file(path, content)
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
    Runs a prediction over the review texts from a Dataframe using a TFIDF model.
    The categories are listed in CLASS_LABELS in the file `common.py`.
    :param model: serialized model
    :param reviews: dataframe
    :return: list of tuples with: file_path, title of review, predicted category for review
    """
    results = []
    for index, review in reviews.iterrows():
        input = common.tok_pos_stem(review["all"])
        labels = model.predict([input])
        label = labels[0]
        results.append([review["file_path"], review["title"], label])
    return results


def predict_cnn(model, df, key_vecs_file, weights_file):
    """
    Runs the predictyion over the review texts from a Dataframe using a CNN model.
    The categories are listed in CLASS_LABELS in the file `common.py`.
    :param model:
    :param reviews: dataframe
    :param key_vecs_file:
    :param weights_file:
    :return:
    """
    results = []

    # builds w2v lexicon
    wv, weights = common.load_word2vec(key_vecs_file, weights_file)
    embedding_vectors = common.create_embedding_vectors(df["all"], wv)

    # converts training data into input format for CNN
    vectors = pad_sequences(embedding_vectors, maxlen=constants.SEQUENCE_DIM, padding='post')

    # runs prediction
    logger.info("predicting...")
    predictions = model.predict(vectors)

    # labels for predictions
    prediction_labels = []
    for prediction in predictions:
        max_one_index = np.argmax(prediction)
        prediction_labels.append(constants.CLASS_LABELS[max_one_index])

    # prepares results
    for index, review in df.iterrows():
        results.append([review["file_path"], review["title"], prediction_labels[index]])
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

    # arguments
    files_to_process = args.file
    model_file = args.model
    outtsv_file = args.outtsv

    # loads reviews
    reviews = load_json_review_from_file(files_to_process)

    # loads model and predicts:
    if model_file.endswith(".h5"):
        # h5 files contain CNN models
        model = load_model(model_file)
        results = predict_cnn(model, reviews, constants.KEY_VECS, constants.WEIGTHS)

    elif model_file.endswith(".pickle"):
        # pickle files contain TDIDF models
        model = joblib.load(model_file)
        results = predict_tfidf(model, reviews)

    else:
        raise Exception("unknown model type")

    logger.info("writing " + outtsv_file)
    write_results_into_tsv_file(results, outtsv_file)

    logger.info("done!")
