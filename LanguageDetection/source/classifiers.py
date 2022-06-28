from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from utils import toNumpyArray
from matplotlib import pyplot as plt 
import seaborn as sns 
import numpy as np
import pandas as pd


def vocabulary_dist(X_train, y_train, X_test):
    '''
    Task: Return a plot for the distribution of words by language in the vocabulary using NB classifier.
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: 

    '''
    print("========")
    print("Computing vocabulary distribution:")
    print("========")
    trainArray = toNumpyArray(X_train)

    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y = clf.feature_log_prob_
    ylab = np.reshape(clf.classes_, (22,1))

    lang_column = np.reshape(np.array([i for i in range(0,22)]), (22,1))
    xmax = np.argmax(y,axis=0)
    lang = np.concatenate((ylab,lang_column), axis=1)

    vocab_count = np.bincount(xmax)
    df = pd.DataFrame({"Language" : lang[:, 0].tolist(), "Number of unigrams": vocab_count.tolist()})
    df=df.sort_values(by= ["Number of unigrams"], ascending=False)
    plt.figure(figsize=(10,6))
    # make barplot
    sns.barplot(x='Language', y="Number of unigrams", data=df)
    plt.xticks(rotation=45)
    plt.show()

# You may add more classifier methods replicating this function
def applyNaiveBayes(X_train, y_train, X_test):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    print("========")
    print("Applying Naive Bayes Classifier")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)

    
    return y_predict


def applyLinearSVC(X_train, y_train, X_test):

    print("========")
    print("Applying Linear Support Vector Classification")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = LinearSVC()
    clf.fit(trainArray, y_train)
    return clf.predict(testArray)


def applyDecisionTree(X_train, y_train, X_test):
    print("========")
    print("Applying Decision Tree Classifier")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = DecisionTreeClassifier()
    clf.fit(trainArray, y_train)
    return clf.predict(testArray)


def applyMultipleLayerPerceptron(X_train, y_train, X_test):
    print("========")
    print("Applying Multiple Layer Perceptron Classifier")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = MLPClassifier()
    clf.fit(trainArray, y_train)
    return clf.predict(testArray)


def applyKNeighborsClassifier(X_train, y_train, X_test):
    print("========")
    print("Applying K-Neighbors Classifier")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = KNeighborsClassifier()
    clf.fit(trainArray, y_train)
    return clf.predict(testArray)


def applyRandomForest(X_train, y_train, X_test):
    print("========")
    print("Applying Random Forest Classifier")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = RandomForestClassifier()
    clf.fit(trainArray, y_train)
    return clf.predict(testArray)


def applyAdaBoost(X_train, y_train, X_test):
    print("========")
    print("Applying AdaBoost Classifier")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = AdaBoostClassifier()
    clf.fit(trainArray, y_train)
    return clf.predict(testArray)


def applyGaussianNB(X_train, y_train, X_test):
    print("========")
    print("Applying Gaussian Naive Bayes Classifier")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = GaussianNB()
    clf.fit(trainArray, y_train)
    return clf.predict(testArray)


def applyQuadraticDiscriminantAnalysis(X_train, y_train, X_test):
    print("========")
    print("Applying Quadratic Discriminant Analysis Classifier")
    print("========")

    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = QuadraticDiscriminantAnalysis()
    clf.fit(trainArray, y_train)
    return clf.predict(testArray)

