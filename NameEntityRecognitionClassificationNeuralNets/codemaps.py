import string
import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataset import *

class Codemaps:
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None):

        if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen)

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen):

        self.maxlen = maxlen
        self.suflen = suflen
        words = set([])
        lc_words = set([])
        sufs = set([])
        labels = set([])
        pos = set([])

        ##REMNANT OF THE CONVOLUTIONAL IMPLEMENTATION
        # extras = set([])

        WordLengths = set([])
        UpperLower = set([])
        FirstCap = set([])
        Syllables = set([])
        HasDash = set([])
        WordShape = set([])
        IsPlural = set([])
        IsInLookup = set([])

        for s in data.sentences():
            for t in s:
                words.add(t['form'])
                lc_words.add(t['lc_form'])
                sufs.add(t['lc_form'][-self.suflen:])
                labels.add(t['tag'])
                pos.add(t['pos'])

                ##REMNANT OF THE CONVOLUTIONAL IMPLEMENTATION
                # for e in t['extra']:
                # extras.add(e)

                WordLengths.add(t['WordLength'])
                UpperLower.add(t['UpperLower'])
                FirstCap.add(t['FirstCap'])
                Syllables.add(t['Syllables'])
                HasDash.add(t['HasDash'])
                WordShape.add(t['WordShape'])
                IsPlural.add(t['IsPlural'])
                IsInLookup.add(t['IsInLookup'])

        self.word_index = {w: i + 2 for i, w in enumerate(list(words))}
        self.word_index['PAD'] = 0  # Padding
        self.word_index['UNK'] = 1  # Unknown words

        self.lc_word_index = {w: i + 2 for i, w in enumerate(list(lc_words))}
        self.lc_word_index['PAD'] = 0  # Padding
        self.lc_word_index['UNK'] = 1  # Unknown words

        self.suf_index = {s: i + 2 for i, s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes

        self.pos_index = {p: i + 2 for i, p in enumerate(list(pos))}
        self.pos_index['PAD'] = 0  # Padding
        self.pos_index['UNK'] = 1  # Unknown suffixes

        self.label_index = {t: i + 1 for i, t in enumerate(list(labels))}
        self.label_index['PAD'] = 0  # Padding

        ##REMNANT OF THE CONVOLUTIONAL IMPLEMENTATION
        # self.extra_index = {t: i+2 for i,t in enumerate(list(extras))}
        # self.extra_index['PAD'] = 0 # Padding
        # self.extra_index['UNK'] = 1  # Unknown suffixes

        self.WordLength_index = {t: i + 2 for i, t in enumerate(list(WordLengths))}
        self.UpperLower_index = {t: i + 1 for i, t in enumerate(list(UpperLower))}
        self.FirstCap_index = {t: i + 1 for i, t in enumerate(list(FirstCap))}
        self.Syllables_index = {t: i + 1 for i, t in enumerate(list(Syllables))}
        self.HasDash_index = {t: i + 1 for i, t in enumerate(list(HasDash))}
        self.WordShape_index = {t: i + 2 for i, t in enumerate(list(WordShape))}
        self.IsPlural_index = {t: i + 1 for i, t in enumerate(list(IsPlural))}
        self.IsInLookup_index = {t: i + 1 for i, t in enumerate(list(IsInLookup))}

        self.WordLength_index['PAD'] = 0  # Padding
        self.WordLength_index['UNK'] = 1  # UNKNOWN
        self.UpperLower_index['PAD'] = 0  # Padding
        self.FirstCap_index['PAD'] = 0  # Padding
        self.Syllables_index['PAD'] = 0  # Padding
        self.HasDash_index['PAD'] = 0  # Padding
        self.WordShape_index['PAD'] = 0  # Padding
        self.WordShape_index['UNK'] = 1  # UNKNOWN
        self.IsPlural_index['PAD'] = 0  # Padding
        self.IsInLookup_index['PAD'] = 0  # Padding

    ## --------- load indexs -----------
    def __load(self, name):
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.lc_word_index = {}
        self.suf_index = {}
        self.label_index = {}
        self.pos_index = {}
        ##REMNANT OF THE CONVOLUTIONAL IMPLEMENTATION
        # self.extra_index = {}
        self.WordLength_index = {}
        self.UpperLower_index = {}
        self.FirstCap_index = {}
        self.Syllables_index = {}
        self.HasDash_index = {}
        self.WordShape_index = {}
        self.IsPlural_index = {}
        self.IsInLookup_index = {}

        with open(name + ".idx") as f:
            for line in f.readlines():
                (t, k, i) = line.split()
                if t == 'MAXLEN':
                    self.maxlen = int(k)
                elif t == 'SUFLEN':
                    self.suflen = int(k)
                elif t == 'WORD':
                    self.word_index[k] = int(i)
                elif t == 'LC_WORD':
                    self.lc_word_index[k] = int(i)
                elif t == 'SUF':
                    self.suf_index[k] = int(i)
                elif t == 'LABEL':
                    self.label_index[k] = int(i)
                elif t == 'POS':
                    self.pos_index[k] = int(i)
                elif t == 'WordLength':
                    self.WordLength_index[k] = int(i)
                elif t == 'UpperLower':
                    self.UpperLower_index[k] = int(i)
                elif t == 'FirstCap':
                    self.FirstCap_index[k] = int(i)
                elif t == 'Syllables':
                    self.Syllables_index[k] = int(i)
                elif t == 'HasDash':
                    self.HasDash_index[k] = int(i)
                elif t == 'WordShape':
                    self.WordShape_index[k] = int(i)
                elif t == 'IsPlural':
                    self.IsPlural_index[k] = int(i)
                elif t == 'IsInLookup':
                    self.IsInLookup_index[k] = int(i)

    ## ---------- Save model and indexs ---------------
    def save(self, name):
        # save indexes
        with open(name + ".idx", "w") as f:
            print('MAXLEN', self.maxlen, "-", file=f)
            print('SUFLEN', self.suflen, "-", file=f)
            for key in self.label_index: print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index: print('WORD', key, self.word_index[key], file=f)
            for key in self.lc_word_index: print('LC_WORD', key, self.lc_word_index[key], file=f)
            for key in self.suf_index: print('SUF', key, self.suf_index[key], file=f)
            for key in self.pos_index: print('POS', key, self.pos_index[key], file=f)

            ##REMNANT OF THE CONVOLUTIONAL IMPLEMENTATION
            # for key in self.extra_index: print('EXTRA', key, self.extra_index[key], file=f)

            for key in self.WordLength_index: print('WordLength', key, self.WordLength_index[key], file=f)
            for key in self.UpperLower_index: print('UpperLower', key, self.UpperLower_index[key], file=f)
            for key in self.FirstCap_index: print('FirstCap', key, self.FirstCap_index[key], file=f)
            for key in self.Syllables_index: print('Syllables', key, self.Syllables_index[key], file=f)
            for key in self.HasDash_index: print('HasDash', key, self.HasDash_index[key], file=f)
            for key in self.WordShape_index: print('WordShape', key, self.WordShape_index[key], file=f)
            for key in self.IsPlural_index: print('IsPlural', key, self.IsPlural_index[key], file=f)
            for key in self.IsInLookup_index: print('IsInLookup', key, self.IsInLookup_index[key], file=f)

    ## --------- encode X from given data -----------
    def encode_words(self, data):
        # encode and pad sentence words
        Xw = [[self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s] for s
              in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index['PAD'])
        # encode and pad sentence lower words
        Xlw = [
            [self.lc_word_index[w['lc_form']] if w['lc_form'] in self.lc_word_index else self.lc_word_index['UNK'] for w
             in s] for s in data.sentences()]
        Xlw = pad_sequences(maxlen=self.maxlen, sequences=Xlw, padding="post", value=self.word_index['PAD'])
        # encode and pad suffixes
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else
               self.suf_index['UNK'] for w in s] for s in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index['PAD'])
        # encode and pad pos-tags
        Xp = [[self.pos_index[w['pos']] if w['pos'] in self.pos_index else self.pos_index['UNK'] for w in s] for s in
              data.sentences()]
        Xp = pad_sequences(maxlen=self.maxlen, sequences=Xp, padding="post", value=self.pos_index['PAD'])

        ##REMNANT OF THE CONVOLUTIONAL IMPLEMENTATION
        # encode and pad extra-tags
        # Xe = [[[self.extra_index[extra] if extra in self.extra_index else self.extra_index['UNK'] for extra in w['extra']] for w in s] for s in data.sentences()]
        # Xe = pad_sequences(maxlen=self.maxlen, sequences=Xe, padding="post", value=self.extra_index['PAD'])

        XWordLength = [[self.WordLength_index[w['WordLength']] if w['WordLength'] in self.WordLength_index else
                        self.WordLength_index['UNK'] for w in s] for s in data.sentences()]
        XWordLength = pad_sequences(maxlen=self.maxlen, sequences=XWordLength, padding="post",
                                    value=self.WordLength_index['PAD'])

        XUpperLower = [[self.UpperLower_index[w['UpperLower']] for w in s] for s in data.sentences()]
        XUpperLower = pad_sequences(maxlen=self.maxlen, sequences=XUpperLower, padding="post",
                                    value=self.UpperLower_index['PAD'])

        XFirstCap = [[self.FirstCap_index[w['FirstCap']] for w in s] for s in data.sentences()]
        XFirstCap = pad_sequences(maxlen=self.maxlen, sequences=XFirstCap, padding="post",
                                  value=self.FirstCap_index['PAD'])

        XHasDash = [[self.HasDash_index[w['HasDash']] for w in s] for s in data.sentences()]
        XHasDash = pad_sequences(maxlen=self.maxlen, sequences=XHasDash, padding="post",
                                 value=self.HasDash_index['PAD'])

        XSyllables = [[self.Syllables_index[w['Syllables']] for w in s] for s in data.sentences()]
        XSyllables = pad_sequences(maxlen=self.maxlen, sequences=XSyllables, padding="post",
                                   value=self.Syllables_index['PAD'])

        XWordShape = [[self.WordShape_index[w['WordShape']] if w['WordShape'] in self.WordShape_index else
                       self.WordShape_index['UNK'] for w in s] for s in data.sentences()]
        XWordShape = pad_sequences(maxlen=self.maxlen, sequences=XWordShape, padding="post",
                                   value=self.WordShape_index['PAD'])

        XIsPlural = [[self.IsPlural_index[w['IsPlural']] for w in s] for s in data.sentences()]
        XIsPlural = pad_sequences(maxlen=self.maxlen, sequences=XIsPlural, padding="post",
                                  value=self.IsPlural_index['PAD'])

        XIsInLookup = [[self.IsInLookup_index[w['IsInLookup']] for w in s] for s in data.sentences()]
        XIsInLookup = pad_sequences(maxlen=self.maxlen, sequences=XIsInLookup, padding="post",
                                    value=self.IsInLookup_index['PAD'])

        # return encoded sequences

        return [Xw, Xlw, Xs, Xp,
                XWordLength,
                XUpperLower,
                XFirstCap,
                XHasDash,
                XSyllables,
                XWordShape,
                XIsPlural,
                XIsInLookup]

    ## --------- encode Y from given data -----------
    def encode_labels(self, data):
        # encode and pad sentence labels
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self):
        return len(self.word_index)

    ## -------- get lower case word index size ---------
    def get_n_lc_words(self):
        return len(self.lc_word_index)

    ## -------- get suf index size ---------
    def get_n_sufs(self):
        return len(self.suf_index)

    ## -------- get label index size ---------
    def get_n_labels(self):
        return len(self.label_index)

    ## -------- get pos index size ---------
    def get_n_pos(self):
        return len(self.pos_index)

    ## -------- get extra index size ---------
    ##REMNANT OF CONVOLUTIONAL IMPLEMENTATION
    # def get_n_extra(self):
    #     return len(self.extra_index)

    def get_n_WordLength(self):
        return len(self.WordLength_index)

    def get_n_UpperLower(self):
        return len(self.UpperLower_index)

    def get_n_FirstCap(self):
        return len(self.FirstCap_index)

    def get_n_HasDash(self):
        return len(self.HasDash_index)

    def get_n_Syllables(self):
        return len(self.Syllables_index)

    def get_n_WordShape(self):
        return len(self.WordShape_index)

    def get_n_IsPlural(self):
        return len(self.IsPlural_index)

    def get_n_IsInLookup(self):
        return len(self.IsInLookup_index)

    ## -------- get index for given word ---------
    def word2idx(self, w):
        return self.word_index[w]

    ## -------- get index for given suffix --------
    def suff2idx(self, s):
        return self.suff_index[s]

    ## -------- get index for given label --------
    def label2idx(self, l):
        return self.label_index[l]

    ## -------- get label name for given index --------
    def idx2label(self, i):
        for l in self.label_index:
            if self.label_index[l] == i:
                return l
        raise KeyError