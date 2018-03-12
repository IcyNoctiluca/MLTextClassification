''' Big Data Assignment         '''
''' Fake News Identification        '''
''' Python 3.6.3        '''


''' importing libs      '''
from bs4 import BeautifulSoup

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM
from keras.layers import Embedding, Dropout, SimpleRNN
from keras.models import Model
from keras.layers.normalization import BatchNormalization

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd
import random
import re
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB


''' defining global variables      '''
RAW_DATASET_ADDR = 'news_ds.csv'                  # must be in same working dir as main.py
GLOVE_EMBEDDINGS_ADDR = 'glove.6B.50d.txt'

LEMMATIZER = WordNetLemmatizer()
PS = PorterStemmer()

MAX_NGRAM_RANGE = 5

MAX_SEQ_LENGTH = 1000
EMBEDDING_DIM = 50
NEURON_PER_LAYER = 280
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCH_COUNT = 8


''' class to hold a sample from dataset ie to hold: id, text and label   '''
class Sample():

    def __init__(self, ID, body, isReal):
        self.id = ID
        self.body = body
        self.isReal = isReal


''' reads data from csv file and returns the three columns      '''
def getData(fileAddr):

    dataset = pd.read_csv(fileAddr)

    ID = dataset.iloc[:, 0].values
    body = dataset.iloc[:, 1].values
    label = dataset.iloc[:, 2].values

    # return array of all samples as objects
    return np.array(
        [Sample( ID[i], body[i], label[i]) for i in range( len(ID) )]
                )


''' writes clean data to csv file      '''
def saveCleanData(samples, outFileName):
    outFileText = 'ID,TEXT,LABEL\n'

    for i in range( len(samples) ):
        outFileText += str(samples[i].id) + ',"'
        outFileText += str(samples[i].body) + '",'
        outFileText += str(samples[i].isReal) + '\n'

    with open(outFileName, 'w') as f:
        f.write(outFileText)


''' checks if a text is in English by seeing if there are English stop words    '''
def isEnglish(text):

    sws = set(stopwords.words('english'))

    # if there are English stop words, then text is English
    for sw in sws:
        if sw in text:
            return True

    # if no English stop words in text, text is probably not English
    return False


''' preprocesses a raw text body depending on parameters  '''
def cleanseText(text, removeStopWords, removeNonLetters, lemmatize, stem):

    # remove new line chars
    text = text.replace('\n', '\r').replace('\r', '')

    # remove all <tags> & make everything lowercase
    text = BeautifulSoup(text).get_text().lower()

    # return None for non-English texts
    if not(isEnglish(text)):
        return None

    # remove all hyperlinks from text
    text = re.sub(r'http\S+', '', text)

    # remove all remaining non-letter characters
    if removeNonLetters:
        text = re.sub('[^a-zA-Z]', ' ', text)
        # this leaves spaces between words as they are

    # remove stop words from text
    if removeStopWords:

        text = ' '.join(
            [word for word in word_tokenize(text)
                if not word in set(stopwords.words('english'))]
                        )

    # convert all words back to root stem
    if stem:

        text = ' '.join(
            [PS.stem(word) for word in word_tokenize(text)]
                        )

    # convert all words back to root lemma
    if lemmatize:

        text = ' '.join(
            [LEMMATIZER.lemmatize(word) for word in word_tokenize(text)]
                        )

    return text


''' cleans all texts in dataset     '''
def cleanseData(samples, removeStopWords, removeNonLetters, lemmatize, stem):

    print ('############################################')
    print ('Cleaning dataset')

    # array to hold cleaned samples
    cleanedSamples = np.array([])

    # iterate through data and cleaning each sample
    for i in range( len(samples) ):
        print ('Cleaning', i+1, 'of', len(samples))

        dirtySample = samples[i]

        # pass text through cleaner function
        cleanedText = cleanseText(
                dirtySample.body, removeStopWords, removeNonLetters, lemmatize, stem)

        # check if cleanseText returned something
        if cleanedText != None:

            cleanSample = Sample(
                    dirtySample.id, cleanedText, dirtySample.isReal
                            )

            # save sample only if it was worth cleaning
            cleanedSamples = np.append(cleanedSamples, cleanSample)

    # shuffle sample array for some extra randomness
    random.shuffle(cleanedSamples)

    return cleanedSamples


''' split data into train/ testing arrays       '''
def trainTestSplit(cleanSamples, testSize):

    random.shuffle(cleanSamples)

    trainSize = 1 - testSize

    partitionIndex = int(len(cleanSamples) * trainSize)

    return cleanSamples[:partitionIndex], cleanSamples[partitionIndex:]


''' flow for Term-Frequency-based classification    '''
def TF_main(NG_RANGE, cleanSamples):

    print ('############################################')
    print ('TF Classification')
    print ('N-Gram range:', NG_RANGE)

    START_TIME = time.time()

    # splitting dataset into training and testing partitions
    trainSamples, testSamples = trainTestSplit(cleanSamples, 0.2)

    # get text body and labels from training samples
    trainTexts = [sample.body for sample in trainSamples]
    trainLabels = [sample.isReal for sample in trainSamples]

    # get text body and labels from testing samples
    testTexts = [sample.body for sample in testSamples]
    testLabels = [sample.isReal for sample in testSamples]

    # setup CV for TF-based predictions
    cv = CountVectorizer(
        max_df=1.0, min_df=1, max_features=None, analyzer='word', ngram_range=(1, NG_RANGE)
                    )

    # convert texts into vectorised features
    trainDataFeatures = cv.fit_transform(trainTexts)
    testDataFeatures = cv.transform(testTexts)

    # fitting the classifier with training data
    clf = MultinomialNB()
    clf.fit(trainDataFeatures, trainLabels)

    # accuracy on the testing data
    print ('Accuracy:')
    print (clf.score(testDataFeatures, testLabels))
    print ()

    # metrics for testing data
    print ('Classification report:')
    print (classification_report(testLabels, clf.predict(testDataFeatures)))
    print ()

    # computation time
    print ('Computation time:')
    print (time.time() - START_TIME)
    print ()


''' flow for Inverse-Term-Frequency-based classification    '''
def TFIDF_main(NG_RANGE, cleanSamples):

    print ('############################################')
    print ('TF-IDF Classification')
    print ('N-Gram range:', NG_RANGE)

    START_TIME = time.time()

    # splitting dataset into training and testing partitions
    trainSamples, testSamples = trainTestSplit(cleanSamples, 0.2)

    # get text body and labels from training samples
    trainTexts = [sample.body for sample in trainSamples]
    trainLabels = [sample.isReal for sample in trainSamples]

    # get text body and labels from testing samples
    testTexts = [sample.body for sample in testSamples]
    testLabels = [sample.isReal for sample in testSamples]

    # setup CV for TFIDF-based predictions
    cv = TfidfVectorizer(
        max_df=1.0, min_df=1, max_features=None, analyzer='word', ngram_range=(1, NG_RANGE)
                    )

    # convert texts into vectorised features
    trainDataFeatures = cv.fit_transform(trainTexts)
    testDataFeatures = cv.transform(testTexts)

    # fitting the classifier with training data
    clf = MultinomialNB()
    clf.fit(trainDataFeatures, trainLabels)

    # accuracy on the testing data
    print ('Accuracy:')
    print (clf.score(testDataFeatures, testLabels))
    print ()

    # metrics for testing data
    print ('Classification report:')
    print (classification_report(testLabels, clf.predict(testDataFeatures)))
    print ()

    # computation time
    print ('Computation time:')
    print (time.time() - START_TIME)
    print ()


''' takes a text as input and returns a list of
    word2vec features using a pre-trained word2vec model      '''
def word2Vec(wordEncodings):

    vocabLength = len(wordEncodings)

    # setup dictionary to hold vectors of each word
    # key of each dictionary entry is the word
    # value is the vector represented by it
    wordVectors = {}

    with open(GLOVE_EMBEDDINGS_ADDR) as f:

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            wordVectors[word] = coefs


    # prepare embedding matrix
    embeddingMatrix = np.zeros((vocabLength + 1, EMBEDDING_DIM))

    for word, encoding in wordEncodings.items():

        embeddingVector = wordVectors.get(word)
        if not(embeddingVector is None):

            embeddingMatrix[encoding] = embeddingVector

    return embeddingMatrix


''' using a LSTM model to classify texts and discriminate
    against fake news                                          '''
def LSTM_main(cleanSamples):

    print ('############################################')
    print ('LSTM neural net')

    START_TIME = time.time()

    # splitting dataset into training and testing partitions
    trainSamples, testSamples = trainTestSplit(cleanSamples, 0.2)

    # getting all texts, training texts and testing texts
    allTexts = np.array([s.body for s in cleanSamples])
    trainTexts = np.array([s.body for s in trainSamples])
    testTexts = np.array([s.body for s in testSamples])

    # same thing for labels
    trainLabels = np.array([s.isReal for s in trainSamples])
    testLabels = np.array([s.isReal for s in testSamples])

    # apply keras tokenizer to entire corpus
    tok = Tokenizer()
    tok.fit_on_texts(allTexts)
    wordEncodings = tok.word_index

    # convert training and testing texts into padded sequences
    trainSequences = pad_sequences(tok.texts_to_sequences(trainTexts), maxlen=MAX_SEQ_LENGTH)
    testSequences = pad_sequences(tok.texts_to_sequences(testTexts), maxlen=MAX_SEQ_LENGTH)

    # gets the vector representation of each word
    # each row is the vector indexed by the dictionary of encodings
    embeddingMatrix = word2Vec(wordEncodings)

    # setup the embedding layer with the matrix
    embeddingLayer = Embedding(
        np.shape(embeddingMatrix)[0], np.shape(embeddingMatrix)[1],
        weights=[embeddingMatrix], input_length=MAX_SEQ_LENGTH, trainable=False
                            )

    # define the model architecture
    inputLayer = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
    embeddedInput = embeddingLayer(inputLayer)

    # first LSTM layer
    lstmLayer1 = LSTM(
        NEURON_PER_LAYER, dropout=DROPOUT,
        recurrent_dropout=DROPOUT, return_sequences=True)

    merged = lstmLayer1(embeddedInput)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    # second LSTM layer
    lstmLayer2 = LSTM(
        NEURON_PER_LAYER, dropout=DROPOUT, recurrent_dropout=DROPOUT)

    merged = lstmLayer2(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    # output layer
    preds = Dense(1, activation='sigmoid')(merged)

    # aggregate
    model = Model(inputs=[inputLayer], outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['acc'])

    # train the model
    model.fit(
        trainSequences, trainLabels, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
        validation_data=(testSequences, testLabels)
            )

    # evaluate the models performance
    print ('Accuracy:')
    _, acc = model.evaluate(testSequences, testLabels, batch_size=BATCH_SIZE)
    print (acc)
    print ()

    # metrics for testing data
    print ('Classification report:')
    print (classification_report(testLabels, np.where(model.predict(testSequences) > 0.5, 1, 0)))
    print ()

    # computation time
    print ('Computation time:')
    print (time.time() - START_TIME)
    print ()


''' using a RNN model to classify texts and discriminate
    against fake news                                          '''
def RNN_main(cleanSamples):

    print ('############################################')
    print ('RNN')

    START_TIME = time.time()

    # splitting dataset into training and testing partitions
    trainSamples, testSamples = trainTestSplit(cleanSamples, 0.2)

    # getting all texts, training texts and testing texts
    allTexts = np.array([s.body for s in cleanSamples])
    trainTexts = np.array([s.body for s in trainSamples])
    testTexts = np.array([s.body for s in testSamples])

    # same thing for labels
    trainLabels = np.array([s.isReal for s in trainSamples])
    testLabels = np.array([s.isReal for s in testSamples])

    # apply keras tokenizer to entire corpus
    tok = Tokenizer()
    tok.fit_on_texts(allTexts)
    wordEncodings = tok.word_index

    # convert training and testing texts into padded sequences
    trainSequences = pad_sequences(tok.texts_to_sequences(trainTexts), maxlen=MAX_SEQ_LENGTH)
    testSequences = pad_sequences(tok.texts_to_sequences(testTexts), maxlen=MAX_SEQ_LENGTH)

    # gets the vector representation of each word
    # each row is the vector indexed by the dictionary of encodings
    embeddingMatrix = word2Vec(wordEncodings)

    # setup the embedding layer with the matrix
    embeddingLayer = Embedding(
        np.shape(embeddingMatrix)[0], np.shape(embeddingMatrix)[1],
        weights=[embeddingMatrix], input_length=MAX_SEQ_LENGTH, trainable=False
                            )

    # define the model architecture
    inputLayer = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
    embeddedInput = embeddingLayer(inputLayer)

    # first RNN layer
    rnnLayer1 = SimpleRNN(
        NEURON_PER_LAYER, dropout=DROPOUT, return_sequences=True)

    merged = rnnLayer1(embeddedInput)
    merged = Dropout(DROPOUT)(merged)

    # second RNN layer
    rnnLayer2 = SimpleRNN(
        NEURON_PER_LAYER, dropout=DROPOUT)

    merged = rnnLayer2(merged)
    merged = Dropout(DROPOUT)(merged)

    # output layer
    preds = Dense(1, activation='sigmoid')(merged)

    # aggregate
    model = Model(inputs=[inputLayer], outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['acc'])

    # train the model
    model.fit(
        trainSequences, trainLabels, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
        validation_data=(testSequences, testLabels)
            )

    # evaluate the models performance
    print ('Accuracy:')
    _, acc = model.evaluate(testSequences, testLabels, batch_size=BATCH_SIZE)
    print (acc)
    print ()

    # metrics for testing data
    print ('Classification report:')
    print (classification_report(testLabels, np.where(model.predict(testSequences) > 0.5, 1, 0)))
    print ()

    # computation time
    print ('Computation time:')
    print (time.time() - START_TIME)
    print ()


''' main flow of programme      '''
def main():

    # loading the dataset
    samples = getData(RAW_DATASET_ADDR)

    # clean the data
    cleanSamples = cleanseData(samples, True, True, False, False)

    # compute shallow learning implementations over full range of N-Grams (1-5)
    for NG_RANGE in range(1, MAX_NGRAM_RANGE+1):

        TF_main(NG_RANGE, cleanSamples)
        TFIDF_main(NG_RANGE, cleanSamples)

    # deep learning implementations
    LSTM_main(cleanSamples)
    RNN_main(cleanSamples)


if __name__ == '__main__':

    main()
