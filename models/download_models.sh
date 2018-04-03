#!/bin/bash
curl http://nlp.franciscodias.pt/repo/bunch/wv/word.vectors -o wv/word.vectors
curl http://nlp.franciscodias.pt/repo/bunch/wv/word.vectors.vectors.npy -o wv/word.vectors.vectors.npy
curl http://nlp.franciscodias.pt/repo/bunch/cnn.h5 -o cnn.h5
curl http://nlp.franciscodias.pt/repo/bunch/svm_stem_stop.pickle -o svm_stem_stop.pickle
curl http://nlp.franciscodias.pt/repo/bunch/svm_pos_stem.pickle -o svm_pos_stem.pickle
