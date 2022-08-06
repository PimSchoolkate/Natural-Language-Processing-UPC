#! /usr/bin/python3


import sys
import random
from contextlib import redirect_stdout

from tensorflow.keras import regularizers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPool1D, Reshape, Concatenate, Flatten, Bidirectional, LSTM

from dataset import *
from codemaps import *


BASE_DIR = "drive/MyDrive/06-DDI-nn"
MODELS_DIR = BASE_DIR + '/models/'
TRAIN_DIR = BASE_DIR
VALIDATION_DIR = BASE_DIR
MODEL_NAME = 'finalmodel'


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



def build_network(idx):
   # sizes
   n_words = codes.get_n_words()
   n_lc_words = codes.get_n_lc_words()
   max_len = codes.maxlen
   n_labels = codes.get_n_labels()
   n_pos = codes.get_n_pos()

   # word input layer & embeddings
   inptW = Input(shape=(max_len,))
   inptLW = Input(shape=(max_len,))

   emb_dim = 200
   embW = Embedding(input_dim=n_words, output_dim=emb_dim,
                    input_length=max_len, embeddings_initializer=Constant(create_embeddings(codes.word_index, emb_dim)),
                    mask_zero=False)(inptW)
   embLW = Embedding(input_dim=n_lc_words, output_dim=emb_dim,
                     input_length=max_len,
                     embeddings_initializer=Constant(create_embeddings(codes.lc_word_index, emb_dim)), mask_zero=False)(
      inptLW)
   # embW = Embedding(input_dim=n_words, output_dim=emb_dim,
   #                   input_length=max_len, mask_zero=False)(inptW)
   # embLW = Embedding(input_dim=n_lc_words, output_dim=emb_dim,
   #                  input_length=max_len, mask_zero=False)(inptLW)

   # dropW = Dropout(0.2)(embW)
   # dropLW = Dropout(0.2)(embLW)
   conc = concatenate([embW, embLW])

   conv = Conv1D(filters=30, kernel_size=3, strides=1, activation='relu', padding='same')(conc)
   # flat= Flatten()(conv)

   # Extra Layers
   # pooling = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv)
   # conv2 = Conv1D(filters=10, kernel_size=5, strides=1, activation='relu', padding='same')(pooling)
   # pooling2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv2)

   # lstm1 = LSTM(units=300)(conv)
   lstm = Bidirectional(LSTM(units=300))(conv)
   # lstm = LSTM(units=300)(drops)

   dense1 = Dense(100, activation=LeakyReLU(alpha=0.1))(lstm)
   drop = Dropout(0.2)(dense1)

   # flat= Flatten()(lstm1)
   out = Dense(n_labels, activation="softmax")(drop)

   model = Model([inptW, inptLW], out)
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   return model


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --


# directory with files to process
trainfile = TRAIN_DIR + '/train.pck'
validationfile = VALIDATION_DIR + '/devel.pck'
modelname = MODEL_NAME

# load train and validation data
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)

# create indexes from training data
max_len = 150
suf_len = 5
codes = Codemaps(traindata, max_len)

# build network
model = build_network(codes)
with redirect_stdout(sys.stderr):
   model.summary()

plotsfile = BASE_DIR + '/' + MODEL_NAME + '.png'

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

