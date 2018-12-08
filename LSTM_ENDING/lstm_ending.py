import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import scipy
from scipy.stats import spearmanr
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from gensim import corpora
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
import gensim
import csv
numpy.random.seed(7)
"""
Read from file and transform it to the relevant LSTM-bow format
"""

class ToLSTM:
    trainSentences = []
    validSentences = []
    trainLabels = None
    validLabels = None

    def __init__(self, trainfilename, testfilename):
        df1=pd.read_csv(trainfilename)
        df2=pd.read_csv(testfilename)
        self.trainS = df1['sentence'].values.tolist()
        self.validS = df2['sentence'].values.tolist()
        self.trainLabels = df1['isEnding'].values.tolist()
        self.validLabels = df2['isEnding'].values.tolist()
        #delete stopchars/words
        print("---Szovegek filterezese elkezdodott---")
        for x in range(0,len(self.trainS)):
            if(self.filter(self.trainS[x])):
                self.trainS[x] = self.filter(self.trainS[x])
            else:
                self.trainS[x] = ['empty', 'sentence']
            
        print("---Tanitoadatok filterezve---")
        for x in range(0,len(self.validS)):
            self.validS[x] = self.filter(self.validS[x])
        print("---Validacios adatok filterezve---")
        trainDictionary = corpora.Dictionary(self.trainS)
        trainCorpus = [trainDictionary.doc2bow(text) for text in self.trainS]
        print("HOSSZ: " + str(len(trainCorpus)))
        for x in trainCorpus:
            if x:
                self.trainSentences.append(self.processBow(x))
            else:
                self.trainSentences.append(1)
        #------------------
        validDictionary = corpora.Dictionary(self.validS)
        validCorpus = [validDictionary.doc2bow(text) for text in self.validS]
        print("HOSSZ: " + str(len(validCorpus)))
        for x in validCorpus:
            if x:
                self.validSentences.append(self.processBow(x))
            else:
                self.validSentences.append(1)


    def processBow(self, str1):
        
        lst = []
        for x in str1: #egy lista elemei
            pair = 0
            for y in x:
                pair += 1
                if pair % 2 == 1:
                    lst.append(y)
        return lst

    def deleteStopChars(self, str): #delete stop chars from test cases
        stopchars = ".,$?!€;-:/%(){}[]^`\'\\\"" #stop chars
        newlist = [] #list of the filtered sentence
        for x in range(0,len(str)): #iterating through a sentence
            willFlag = 0
            notFlag = 0
            wouldFlag = 0
            haveFlag = 0
            newstr = ""
            apostr = 0
            processedStr=""
            #deleting contractions
            if "\'s" in str[x]:
                processedStr = str[x].replace("\'s","")
            elif "\'ll" in str[x]:
                processedStr = str[x].replace("\'ll","")
                willFlag = 1
            elif "n\'t" in str[x]:
                processedStr = str[x].replace("n\'t","")
                notFlag = 1
            elif "\'d" in str[x]:
                processedStr = str[x].replace("\'d","")
                wouldFlag = 1
            elif "\'ve" in str[x]:
                processedStr = str[x].replace("\'ve","")
                haveFlag = 1
            elif "\'m" in str[x]:
                processedStr = str[x].replace("\'m","")
                haveFlag = 1
            elif "o\'" in str[x]:
                processedStr = str[x].replace("o\'","")
                haveFlag = 1

            else:
                processedStr = str[x]
            for y in range(0,len(processedStr)): #iterating through a word
                if processedStr[y] not in stopchars: #if a char not in the stoplist it can go forward
                    newstr += processedStr[y] #adding to the filtered word
            newlist.append(newstr) #add the filtered word to a sentence
        return newlist #return with the sentence

    def filter(self, sentence):
        """
        This method filters a given sentence in every possible way
        """
        stop_words = set(stopwords.words('english')) #nltk stopwords
        stop_words.add("would")
        stop_words.add("cannot")
        sentence = sentence.lower().split() #lowercase all of the words in the dataset
        sentence = self.deleteStopChars(sentence) #filtering to stopchars
        sentence = [w for w in sentence if w not in stop_words] #filtering to stopwords
        return sentence #return with a filtered sentence

makeObj = ToLSTM('end_or_not_train_data.csv','val_isEnding.csv')


top_words = 500000 #amennyi szot megtartunk a halmazbol


max_review_length = 1

#Egyforma hosszuva tesszuk (?????? elvileg) a kiertekeleshez

embedding_vecor_length = 100 #a beagyazasi vektorok hossza(szavak)
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

print(len((makeObj.validSentences)))
X_valid = sequence.pad_sequences(makeObj.validSentences, maxlen=max_review_length)
#for x in makeObj.trainSentences:
    #print(x)
X_train = sequence.pad_sequences(makeObj.trainSentences, maxlen=max_review_length)
print("train data hossza: " + str(len(X_train)) + " es a train labelek hossza: " + str(len(makeObj.trainLabels)))
print("valid data hossza: " + str(len(X_valid)) + " es a valid labelek hossza: " + str(len(makeObj.validLabels)))
#print(makeObj.validSentences)
#print(X_valid)


model.add(LSTM(100)) #100 memoriaegyseg, okos neuronok
model.add(Dense(1, activation='sigmoid')) #hozzáadunk egy dense layert, mivel binaris osztalyozas, ez fog donteni a sigmoid fgv-nyel
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, makeObj.trainLabels, validation_data=(X_valid, makeObj.validLabels), epochs=3, batch_size=64)
scores = model.evaluate(X_valid, makeObj.validLabels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))