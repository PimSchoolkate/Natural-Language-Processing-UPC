from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
import joblib

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
            print("No model was initialized, please select a mode ('TRAIN', 'PREDICT')")

    def define_model(self, mod):
        self._vectorizer = DictVectorizer()
        if mod == 'DT':
            self._model = DecisionTreeClassifier(max_depth=70)
        elif mod == 'SVM':
            self._model = LinearSVC(max_iter=2500)
        elif mod == "NB":
            self._model = MultinomialNB()
        else:
            print("Model does not exist")

    def train(self, x, y):
        self._fit_vectorizer(x)
        X = self.vectorize(x)
        print("Training model...")
        if self.mod in ('DT', 'SVM', 'NB'):
            self._model.fit(X, y)
        else:
            print("This model has not been defined for the trainer")
        print(f"{self.mod} has been trained!")

    def predict(self, x):
        X = self.vectorize(x)
        return self._model.predict(X)

    def _fit_vectorizer(self, x):
        self._vectorizer.fit(x)

    def vectorize(self, x):
        return self._vectorizer.transform(x).toarray()

    def pickle_model_and_encoder(self, name):
        joblib.dump(self._model, f"./models/{name}-mod.joblib")
        joblib.dump(self._vectorizer, f"./models/{name}-enc.joblib")
        # with open(name + '.model', 'wb') as file:
        #     pickle.dump(self._model, file)
        # with open(name + '.enc', 'wb') as file:
        #     pickle.dump(self._vectorizer, file)

    def unpickle_model_and_encoder(self, name):
        self._model = joblib.load(f"./models/{name}-mod.joblib")
        self._vectorizer = joblib.load(f"./models/{name}-enc.joblib")

        # with open(name + '.model', 'rb') as file:
        #     self._model_oh = pickle.load(file)
        # with open(name + '.enc', 'rb') as file:
        #     self._vectorizer = pickle.load(file)

