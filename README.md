<p align="center">
  <img height="100px" src="https://bunch.ai/wp-content/themes/bunch/images/bunch-logo-rgb.svg" alt="Bunch" />
</p>

# Datascience Challenge

## Intro

This is my submission to the Bunch.ai code challenge.
The objective of this challenge is:

+ "to **build a text classifier that predicts to which dimension a review is the most related to**, based on its content".

This submission contains e files:

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

## How to install

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

## How to run

There are 2 runnable scripts:

+ **training**: 
    + `python training.py [MODEL FILE] [OUTPUT TSV] [JSON FILES...]`

+ **prediction**:
    + `python predict.py [DATAFRAME FILE] (tfidf|convnet) [MODEL FILE]`

## Answers to the Requirements

Answers to the requirements of this challenge:

* Use Python as programming language
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) done!
* Achieve an accuracy of 90% of trained classifier
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) >90% acc during
    training-validation (lower accuracy using the test dataset: ~ 46% CNN ~ 56% TF-IDF), done!
* Process the unlabelled data using the classifier and save the output in a file, so that we can analyze it
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `predict.py` does
    the job, done!
* Explain your approach: your code should be documented and you should be able to
explain your decisions
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **notebook** + **code**, done!
* Discuss the next steps to further improve the classifier (eg: work with multiple
languages)
    
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+)  **more data** :)
        
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **tunning classifiers**, choose the best hyper-parameters, test other
    implementations
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **multiple languages**: language detection, use a model for each language,
        however may exist texts with more than one language and also foreign expressions
        that can be used in some contexts ("download", "performance", "laisse-faire")
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **understand the text**, improving the classification dealing with some
    characteristics of Natural Language such as ambiguities, intensifiers, recursions (stating
    different objects such as "the company which department which employees
    which computers ..."), sarcasm (I guess Glassdoor is full of this), typos ("tpyos"),
    acronyms (QA, QR, HR, AAR,...), among others.
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **find the regions of interest inside the text**, instead of classifying
    a sentence as a whole, find the regions of the text that convey more information
    and assign weights to each one; this way, the NLP tasks and classification tasks
    could be detached from each other;
* Explain how would make this classifier available in production
    + ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) I could list some possibilities
    for discussion, each one with some pros and cons:
    
        + **deploy as webapp** using a gateway and a framework such as Bottle or
    Flask; when the application is started, the model is also loaded
        into memory, improving the performance;
        
        + **deploy as microservice** using also a gateway and a message broker
    such as RabbitMQ; this way we could improve the availability using
    several free microservices subscribing the same type of messages;
        + **deploy in the cloud** using Cloud ML or Amazon SageMaker;
        improves the performance, availability, and reduces the potential time
        for VM/cluster administration.

## Models

Files containing the models were not uploaded to **git**. Please use the provided
download script.

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
