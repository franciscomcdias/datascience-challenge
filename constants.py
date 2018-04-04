CLASS_LABELS = ['adaptability', 'collaboration', 'customer', 'detail', 'integrity', 'result']

# paths to the word2vec model
KEY_VECS = "models/wv/word.vectors"
WEIGTHS = "models/wv/word.vectors.vectors.npy"

# convnet parameters
DROPOUT_VAL = 0.1
CLASSES_DIM = 6
SEQUENCE_DIM = 200

# titles of the sehments
SEGMENT_TITLES = '(Pros |Cons |Advice to Management )'

# POS allowed when filtering words by POS
ALLOWED_POS = ["NN", "NNS", "JJ", "VB", "VBG"]
