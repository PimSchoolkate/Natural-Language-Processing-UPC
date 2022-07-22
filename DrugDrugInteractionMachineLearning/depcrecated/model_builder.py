from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import pickle

class model():
    def __init__(self, mode, mod=None, name=None):
        if mode == "TRAIN":
            if mod is None:
                raise ValueError("Please define the type of model in mod")
            self.mod = mod
            self.define_model(mod)
        elif mode == "PREDICT":
            if name is None:
                raise ValueError("Please define the name of the model in name")
            self.unpickle_model_and_encoder(name)
        else:
            print("No model was initialized, please select a mode (TRAIN, PREDICT)")

    def  define_model(self, mod):
        self._label_encoder = LabelEncoder()
        self._onehot = OneHotEncoder()
        if mod == 'DT':
            self._model_oh = DecisionTreeClassifier(max_depth=10)
            self._model_le = DecisionTreeClassifier(max_depth=10)
        elif mod == 'SVM':
            self._model_oh = LinearSVC()
            self._model_le = LinearSVC()
        elif mod == "NB":
            self._model_oh = MultinomialNB()
            self._model_le = MultinomialNB()
        else:
            print("Model does not exist")

    def train(self, x, y):
        self._fit_encoders(x)
        x_oh = self._one_hot(x)
        x_le = self._label_encode(x)
        print("Training model...")
        if self.mod in ('DT', 'SVM'):
            self._model_oh.fit(x_oh, y)
            self._model_le.fit(x_le, y)
        else:
            print("This model has not been defined for the trainer")
        print(f"{self.mod} has been trained!")

    def predict(self, x):
        x_oh = self._one_hot(x)
        x_le = self._label_encode(x)
        return self._model_oh.predict(x_oh), self._model_le.predict(x_le)

    def vectorize(self, x):
        self._vectorizer = DictVectorizer()
        self._vectorizer.fit_transform(x)

    def _fit_encoders(self, xvalues:list):
        print("Training encoder...")
        values = get_unique_values(xvalues)
        self._label_encoder.fit(values)
        self._onehot.fit(xvalues)
        print("Encoder has been trained!")

    def _one_hot(self, xvalues:list):
        print("One Hot encoding data...")
        return self._onehot.transform(xvalues).toarray()

    def _label_encode(self, xvalues:list):
        print("Label Encoding data...")
        x = np.array([self._label_encoder.transform(samp) for samp in pd.DataFrame(xvalues).astype(str).values])
        return x

    def pickle_model_and_encoder(self, name):
        with open(name + '-oh.model', 'wb') as file:
            pickle.dump(self._model_oh, file)
        with open(name + '-le.model', 'wb') as file:
            pickle.dump(self._model_le, file)
        with open(name + "-oh.enc", 'wb') as file:
            pickle.dump(self._onehot, file)
        with open(name + "-le.enc", 'wb') as file:
            pickle.dump(self._label_encoder, file)

    def unpickle_model_and_encoder(self, name):
        with open(name + '-oh.model', 'rb') as file:
            self._model_oh = pickle.load(file)
        with open(name + '-le.model', 'rb') as file:
            self._model_le = pickle.load(file)
        with open(name + "-oh.enc", 'rb') as file:
            self._onehot = pickle.load(file)
        with open(name + "-le.enc", 'rb') as file:
            self._label_encoder = pickle.load(file)



def get_unique_values(nested_lists:list):
    uniques = set()
    get_uniques(uniques, nested_lists)
    return list(uniques)


def get_uniques(_u: set, _l: list):
    if type(_l) == type(list()):
        for _i in _l:
            get_uniques(_u, _i)
    else:
        _u.add(_l)
