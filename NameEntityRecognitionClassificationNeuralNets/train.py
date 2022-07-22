#! /usr/bin/python3

import sys
from contextlib import redirect_stdout

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, Lambda, Constant, LeakyReLU

import numpy as np

from dataset import *
from codemaps import *

def create_embeddings(word_index, emb_dim):
  embeddings_index = {}
  with open(BASE_DIR + f"/Embeddings/glove.6B.{emb_dim}d.txt") as f:
      for line in f:
          word, coefs = line.split(maxsplit=1)
          coefs = np.fromstring(coefs, "f", sep=" ")
          embeddings_index[word] = coefs

  print("Found %s word vectors." % len(embeddings_index))

  num_tokens = len(word_index)
  hits = 0
  misses = 0

  # Prepare embedding matrix
  embedding_matrix = np.zeros((num_tokens, emb_dim))
  for word, i in word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # Words not found in embedding index will be all-zeros.
          # This includes the representation for "padding" and "OOV"
          embedding_matrix[i] = embedding_vector
          hits += 1
      else:
          misses += 1
  print("Converted %d words (%d misses)" % (hits, misses))
  return embedding_matrix


def build_network(codes):
    # sizes
    n_words = codes.get_n_words()
    n_sufs = codes.get_n_sufs()
    n_labels = codes.get_n_labels()
    n_pos = codes.get_n_pos()
    n_lc_words = codes.get_n_lc_words()
    #  n_extra = codes.get_n_extra()
    max_len = codes.maxlen
    emb_dim = 100

    inptW = Input(shape=(max_len,))  # word input layer & embeddings
    embW = Embedding(input_dim=n_words,
                     output_dim=emb_dim,
                     input_length=max_len,
                     embeddings_initializer=Constant(create_embeddings(codes.word_index, emb_dim)),
                     mask_zero=True)(inptW)

    inptLW = Input(shape=(max_len,))  ## Lower case word input layer & embeddings
    embLW = Embedding(input_dim=n_lc_words,
                      output_dim=emb_dim,
                      input_length=max_len,
                      embeddings_initializer=Constant(create_embeddings(codes.lc_word_index, emb_dim)),
                      mask_zero=True)(inptLW)

    inptS = Input(shape=(max_len,))  # suf input layer & embeddings
    embS = Embedding(input_dim=n_sufs, output_dim=50,
                     input_length=max_len, mask_zero=True)(inptS)

    inptP = Input(shape=(max_len,))  # pos input layer & embeddings
    embP = Embedding(input_dim=n_pos, output_dim=50, input_length=max_len,
                     mask_zero=True)(inptP)

    ## ADDING EXTRA FEATURES STARTS HERE...
    ## They are all commented out in order to test different configurations..
    ## could be more optimal with gridsearch config...

    # Lenght of the word
    #  inptWordLength = Input(shape=(max_len,))
    #  embWordLength = Embedding(input_dim=codes.get_n_WordLength(),
    #                            output_dim=5,
    #                            input_length=max_len,
    #                            mask_zero=True)(inptWordLength)
    #  dropWordLength = Dropout(0.1)(embWordLength)

    #  # If the word is in UPPER cases, lower cases, or a CoMbInAtIoN
    #  inptUpperLower = Input(shape=(max_len,))
    #  embUpperLower = Embedding(input_dim=codes.get_n_UpperLower(),
    #                            output_dim=2,
    #                            input_length=max_len,
    #                            mask_zero=True)(inptUpperLower)
    #  dropUpperLower = Dropout(0.1)(embUpperLower)

    #  ## If the first letter of the word is in UPPER case
    #  inptFirstCap = Input(shape=(max_len,))
    #  embFirstCap = Embedding(input_dim=codes.get_n_FirstCap(),
    #                          output_dim=2,
    #                          input_length=max_len,
    #                          mask_zero=True)(inptFirstCap)
    #  dropFirstCap = Dropout(0.1)(embFirstCap)

    ## If the word contains a dash
    #  inptHasDash = Input(shape=(max_len,))
    #  embHasDash = Embedding(input_dim=codes.get_n_HasDash(),
    #                         output_dim=2,
    #                         input_length=max_len,
    #                         mask_zero=True)(inptHasDash)
    #  dropHasDash = Dropout(0.1)(embHasDash)

    ## How many syllables the word has (Aprrox)
    #  inptSyllables = Input(shape=(max_len,))
    #  embSyllables = Embedding(input_dim=codes.get_n_Syllables(),
    #                           output_dim=5,
    #                           input_length=max_len,
    #                           mask_zero=True)(inptSyllables)
    #  dropSyllables = Dropout(0.1)(embSyllables)

    #  ## The shape of the word
    #  inptWordShape = Input(shape=(max_len,))
    #  embWordShape = Embedding(input_dim=codes.get_n_WordShape(),
    #                           output_dim=50,
    #                           input_length=max_len,
    #                           mask_zero=True)(inptWordShape)
    #  dropWordShape = Dropout(0.1)(embWordShape)

    #  ## If the word is a plurl (only checking the 's')
    #  inptIsPlural = Input(shape=(max_len,))
    #  embIsPlural = Embedding(input_dim=codes.get_n_IsPlural(),
    #                          output_dim=2,
    #                          input_length=max_len,
    #                          mask_zero=True)(inptIsPlural)
    #  dropIsPlural = Dropout(0.1)(embIsPlural)

    ## If the word is in the drug-bank file
    #  inptIsInLookup = Input(shape=(max_len,))
    #  embIsInLookup = Embedding(input_dim=codes.get_n_IsInLookup(),
    #                            output_dim=2,
    #                            input_length=max_len,
    #                            mask_zero=True)(inptIsInLookup)
    #  dropIsInLookup = Dropout(0.1)(embIsInLookup)

    # Concatenate all of them together here
    #  dropExtra = concatenate([dropWordLength,
    #                           dropWordShape,
    #                           dropFirstCap,
    #                           dropUpperLower,
    #                           dropHasDash,
    #                           dropSyllables,
    #                           dropIsPlural,
    #                           dropIsInLookup])

    ## Convolutional Solution to the extra parameters
    ## REMNANTS OF THE CONVOLUTIONAL IMPLEMENTATION

    #  inptE = Input(shape=(max_len,7,))
    #  embE = Embedding(input_dim=n_extra, output_dim=50, input_length=max_len,
    #                   mask_zero=True)(inptE)
    #  reshE = Reshape((150,50,7))(embE)
    #  convE = Conv1D(filters=5,kernel_size=3, padding='same',
    #                 activation=LeakyReLU(alpha=0.1))(reshE)
    #  convE = Conv1D(filters=3,kernel_size=3, padding='same',
    #                 activation=LeakyReLU(alpha=0.1))(convE)
    #  convE = Conv1D(filters=1,kernel_size=3, padding='same',
    #                 activation=LeakyReLU(alpha=0.1))(convE)
    #  reshE = Reshape((150, 50))(convE)

    dropW = Dropout(0.1)(embW)
    dropLW = Dropout(0.1)(embLW)
    dropS = Dropout(0.1)(embS)
    dropP = Dropout(0.1)(embP)
    #  dropE = Dropout(0.1)(reshE)
    drops = concatenate([dropW, dropLW, dropS, dropP,
                         # dropExtra
                         ])

    # biLSTM
    bilstm = Bidirectional(LSTM(units=300,
                                return_sequences=True))(drops)
    # output softmax layer

    dense1 = Dense(100, activation=LeakyReLU(alpha=0.1))(bilstm)
    drop1 = Dropout(0.1)(dense1)

    out = TimeDistributed(Dense(n_labels, activation="softmax"))(drop1)

    # build and compile model

    model = Model([inptW, inptLW, inptS, inptP,
                   # inptWordLength,
                   # inptUpperLower,
                   # inptFirstCap,
                   # inptHasDash,
                   # inptSyllables,
                   # inptWordShape,
                   # inptIsPlural,
                   # inptIsInLookup
                   ], out)
    model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

# directory with files to process
traindir = TRAIN_DIR
validationdir = VALIDATION_DIR

# load train and validation data
traindata = Dataset(traindir)
valdata = Dataset(validationdir)

# create indexes from training data
max_len = 150
suf_len = 6
codes = Codemaps(traindata, max_len, suf_len)


# build network
model = build_network(codes)
with redirect_stdout(sys.stderr):
    model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# train model
with redirect_stdout(sys.stderr):
    model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv, Yv), verbose=1)

# save model and indexs
model.save(MODELS_DIR + MODEL_NAME)
codes.save(MODELS_DIR + MODEL_NAME)

