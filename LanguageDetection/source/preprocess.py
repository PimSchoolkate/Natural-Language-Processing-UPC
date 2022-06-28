import nltk
import string
import pandas as pd

#Tokenizer function. You can add here different preprocesses.
def preprocess(sentences, labels, stemming=True, lemmatizing=True, remove_stopwords=True):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    # Place your code here
    # Keep in mind that sentence splitting affects the number of sentences
    # and therefore, you should replicate labels to match.

    preprocessed = pd.Series([])
    stop_words = []

    if remove_stopwords == "True":
        stop_words = generateStopwords(labels)

    for sentence in sentences:
        sentence = removePunctuation(sentence)
        sentence = lowerCharacters(sentence)

        if remove_stopwords == "True":
            removeStopwords(sentence, stop_words)

        if stemming == "True":
            stemmer = getPorterStemmer()
            sentence = stemSentence(sentence, stemmer)

        if lemmatizing == "True":
            lemmatizer = getWordNetLemmatizer()
            sentence = lemmatizeSentence(sentence, lemmatizer)

        sentence = pd.Series([sentence])
        preprocessed = pd.concat([preprocessed, sentence], ignore_index=True)
    return preprocessed,labels


def lowerCharacters(sentence):
    return sentence.lower()


def removePunctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


def getWordNetLemmatizer():
    return nltk.stem.WordNetLemmatizer()


def tokenizeSentence(sentence):
    return nltk.tokenize.word_tokenize(sentence)


def getPorterStemmer():
    return nltk.stem.PorterStemmer()


def generateStopwords(labels):
    unique_labels = labels.unique()
    for i in range(len(unique_labels)):
        unique_labels[i] = unique_labels[i].lower()

    common_languages = list(set(unique_labels).intersection(nltk.corpus.stopwords.fileids()))
    stop_words = []
    for i in range(len(common_languages)):
        stop_words.extend(set(nltk.corpus.stopwords.words(common_languages[i])))
    return stop_words


def removeStopwords(sentence, stop_words):
    word_tokens = tokenizeSentence(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return "".join(filtered_sentence)


def stemSentence(sentence, stemmer):
    token_words = tokenizeSentence(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def lemmatizeSentence(sentence, lemmatizer):
    token_words = tokenizeSentence(sentence)
    lemma_sentence = []
    for word in token_words:
        lemma_sentence.append(lemmatizer.lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)



def preprocess_char(sentences):
    preprocessed = pd.Series([])

    for sentence in sentences:
        #sentence = removePunctuation(sentence)
        sentence = lowerCharacters(sentence)

        sentence = pd.Series([sentence])
        preprocessed = pd.concat([preprocessed, sentence], ignore_index=True)

    return preprocessed

