import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from utils import *
from classifiers import *
from preprocess import  preprocess, preprocess_char

seed = 42
random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", 
                        help="Input data in csv format", type=str)
    parser.add_argument("-v", "--voc_size", 
                        help="Vocabulary size", type=int)
    parser.add_argument("-a", "--analyzer",
                         help="Tokenization level: {word, char}", 
                        type=str, choices=['word','char'])
    parser.add_argument("-c", "--classifier",
                        help="Choose which classifier is used, default NB",
                        type=str, choices=["NB", "SVC", "DT", "MLP", "KNN", "RF", "ADA", "GNB", "QDA"],
                        default="NB")
    parser.add_argument("-s", "--stemmer",
                        help="Stemming True or False, default False",
                        type=str, choices=["True", "False"],
                        default="False")
    parser.add_argument("-l", "--lemmatizer",
                        help="lemmatizing True or False, default False",
                        type=str, choices=["True", "False"],
                        default="False")
    parser.add_argument("-w", "--stopwords",
                        help="remove stopwords True or False, default False",
                        type=str, choices=["True", "False"],
                        default="False")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    
    # Languages
    languages = set(raw['language'])
    print('========')
    print('Languages', languages)
    print('========')

    # Split Train and Test sets
    X=raw['Text']
    y=raw['language']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    print('========')
    print('Split sizes:')
    print('Train:', len(X_train))
    print('Test:', len(X_test))
    print('========')
    
    # Preprocess text (Word granularity only)
    print("Stemming : " + str(args.stemmer))
    print("Lemmatizing : " + str(args.lemmatizer))
    print("Remove stopwords : " + str(args.stopwords))


    if args.analyzer == 'word':
        X_train, y_train = preprocess(X_train,y_train, stemming=args.stemmer, lemmatizing=args.lemmatizer,
                                      remove_stopwords=args.stopwords)
        X_test, y_test = preprocess(X_test,y_test, stemming=args.stemmer, lemmatizing=args.lemmatizer,
                                      remove_stopwords=args.stopwords)

    if args.analyzer == 'char':
        X_train = preprocess_char(X_train)
        X_test = preprocess_char(X_test)

    #Compute text features
    features, X_train_raw, X_test_raw = compute_features(X_train, 
                                                            X_test, 
                                                            analyzer=args.analyzer, 
                                                            max_features=args.voc_size)

    print('========')
    print('Number of tokens in the vocabulary:', len(features))
    print('Coverage: ', compute_coverage(features, X_test.values, analyzer=args.analyzer))
    print('========')

   
    #Apply Classifier  
    X_train, X_test = normalizeData(X_train_raw, X_test_raw)

    if args.classifier == "NB":
        y_predict = applyNaiveBayes(X_train, y_train, X_test)
         # Compute vocabulary features in case we are using NB classifier.
        vocabulary_dist(X_train, y_train, X_test)
    if args.classifier == "SVC":
        y_predict = applyLinearSVC(X_train, y_train, X_test)
    if args.classifier == "DT":
        y_predict = applyDecisionTree(X_train, y_train, X_test)
    if args.classifier == "MLP":
        ## takes long to train
        y_predict = applyMultipleLayerPerceptron(X_train, y_train, X_test)
    if args.classifier == "KNN":
        y_predict = applyKNeighborsClassifier(X_train, y_train, X_test)
    if args.classifier == "RF":
        y_predict = applyRandomForest(X_train, y_train, X_test)
    if args.classifier == "ADA":
        y_predict = applyAdaBoost(X_train, y_train, X_test)
    if args.classifier == "GNB":
        y_predict = applyGaussianNB(X_train, y_train, X_test)
    if args.classifier == "QDA":
        y_predict = applyQuadraticDiscriminantAnalysis(X_train, y_train, X_test)


    print('========')
    print('Prediction Results:')    
    plot_F_Scores(y_test, y_predict)
    print('========')
    
    plot_Confusion_Matrix(y_test, y_predict, "Greens") 


    #Plot PCA
    print('========')
    print('PCA and Explained Variance:')
    plotPCA(X_train, X_test,y_test, languages)
    print('========')
