# Bunch's Datascience Challenge

## Introduction

This is my submission to the Bunch.ai code challenge.
The objective of this challenge was to **build a text
classifier that predicts to which dimension a** text **review
is the most related to**, based on its content.

This solution consisted of an NLP multi-class text classification
problem and reading about research on organisational culture profiling.

Each *class* relates with one of the following
company culture *dimension*:

`Adaptability`, 
`Collaboration`, 
`Customer`, 
`Detail`, 
`Integrity`, 
`Result`

The evaluation metric used in this challenge was *accuracy*.
The evaluation of the models showed that a simple CNN classifier
reaches an accuracy of 0.46 and TF-IDF with a SVM 0.56. 

Bunch.ai is a Berlin-based startup that assesses the cultural
profile of companies and teams in order
Read about their awesome solution and blog at the company's
website:
[https://bunch.ai](https://bunch.ai)




## Project Planning

I use a [CRISP](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
approach to build a project planning that can divide this project
into 6 sequential steps:

1. **Understanding the Business**
    + theory of company profiling, solutions for text categorisation, understand the problem
2. **Understanding the Data**
    + structure of the dataset, check quality of the data,adding new sources of data, drawing hypothesis
3. **Preparing the Data**
    + pre-processing the text dataset, creating train/test/eval datasets
4. **Modelling**
    + choose an evaluation metric, training and testing models
5. **Evaluating Hypothesis**
    + evaluate the models' output using a chosen metric, re-iterate
6. **Deploying**
    + final requirements, deploying the solutions to a production environment

## Theory of Organisational Culture

The theory of *company culture* that is presented here is
also referred as O'Reilly model.
The most important theoretical research work in this
area has been developed by Charles O'Reilly and Jennifer
Chatman:

+ https://www.researchgate.net/profile/Charles_OReilly
+ https://www.researchgate.net/profile/Jennifer_Chatman



## Datasets


## Experimental Setup

+ 1 notebook with all the exploratory work and evaluation;
+ 2 files (training.py, predict.py) that implement the same code as the notebook. 


This submission presents 2 solutions for the problem of text classification:

+ TF-IDF + SVM, a classical approach;
+ CNN, a deep learning approach.

Models for solutions were evaluated and
compared using significance tests with *p* < *0.05* .

The results showed that the TF-IDF-based classifiers outperformed the CNNs models
in this setup conditions.

## Delivered files

+ **challenge.ipynb** : jupyter notebook with the exploratory analysis and explanation
    + <p align="center">
        <img height="200px" src="http://nlp.franciscodias.pt/repo/bunch/notebook.png" alt="this notebook" />
    </p>

+ **challenge.html** : HTML version of the jupyter notebook
+ **training.py** : training script
+ **predict.py** : prediction script 
+ **common.py** : code that is shared between the trainer and predicted,
+ **constants.py** : shared constants;
+ **results.tsv** : TSV file containing the category predictions for the unlabelled files.

## How to...

#### How to install

+ use python 3.4+
+ activate your favourite virtual environment
+ install requirements ```pip install -r requirements.txt```
+ run setup: ``python setup.py install`` for downloading NLTK modules
    + or execute it manualy:
        + download the following modules from NLTK:
            + `averaged_perceptron_tagger`
            + `punkt`
+ download the word2vec (required) and sample models (optional) into the folder `models` using the provided script `models/download_models.sh`:
    + run `$ sh download_models.sh`

#### How to run

There are 2 runnable scripts:

+ **training**: 
    `python training.py [MODEL FILE] [OUTPUT TSV] [JSON FILES...]`

+ **prediction**:
    `python predict.py [DATAFRAME FILE] (tfidf|convnet) [MODEL FILE]`

## Examples

Running over all the files inside `data/unlabelled-dataset`:

`find data/unlabelled-dataset -type f -print0 | xargs -r0 python predict.py models/svm_pos_stem.pickle results.tsv`

---

`python training.py data/labelled_dataset.pickle convnet cnn.h5`

```
Using TensorFlow backend.
loading dataframe...
pre-processing dataframe...
	- no categories
	- duplicates
	- non-english
	- pros/cons
loading word2vec model...
creating embedding vectors...
Train on 178 samples, validate on 20 samples
Epoch 1/100
177/177 [==============================] - 2s 9ms/step - loss: 1.8915 - acc: 0.1299 - val_loss: 1.7787 - val_acc: 0.2500
Epoch 2/100
177/177 [==============================] - 1s 6ms/step - loss: 1.7165 - acc: 0.3107 - val_loss: 1.6497 - val_acc: 0.2500
Epoch 3/100
177/177 [==============================] - 1s 6ms/step - loss: 1.6121 - acc: 0.3616 - val_loss: 1.5863 - val_acc: 0.2500
Epoch 4/100
177/177 [==============================] - 1s 6ms/step - loss: 1.5392 - acc: 0.3842 - val_loss: 1.5636 - val_acc: 0.2500
Epoch 5/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4807 - acc: 0.3898 - val_loss: 1.5572 - val_acc: 0.2500
Epoch 6/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4823 - acc: 0.4124 - val_loss: 1.5607 - val_acc: 0.2500
Epoch 7/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4428 - acc: 0.4011 - val_loss: 1.5567 - val_acc: 0.3000
Epoch 8/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4804 - acc: 0.3842 - val_loss: 1.5609 - val_acc: 0.3000
Epoch 9/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4658 - acc: 0.3785 - val_loss: 1.5542 - val_acc: 0.2500
Epoch 10/100
177/177 [==============================] - 1s 6ms/step - loss: 1.4377 - acc: 0.4294 - val_loss: 1.5387 - val_acc: 0.3000
Epoch 11/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4435 - acc: 0.4407 - val_loss: 1.5250 - val_acc: 0.3000
Epoch 12/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4432 - acc: 0.4068 - val_loss: 1.5156 - val_acc: 0.3000
Epoch 13/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4329 - acc: 0.4124 - val_loss: 1.5099 - val_acc: 0.3000
Epoch 14/100
177/177 [==============================] - 1s 7ms/step - loss: 1.4091 - acc: 0.4294 - val_loss: 1.5148 - val_acc: 0.2500


saving cnn.h5
done!
```

---


`python predict.py cnn.h5 output.tsv data/unlabelled-dataset/Huge.json`

```
Using TensorFlow backend.
loading word2vec model...
creating embedding vectors...
predicting...
writing output.tsv
done!
```

```
data/unlabelled-dataset/Huge.json	Visual Designer	collaboration	
```


