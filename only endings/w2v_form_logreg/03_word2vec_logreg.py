import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import scipy
from scipy.stats import spearmanr
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
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

class convertToVec:

    model = None
    resultFileName = None
    sentenceList = []
    vectorList = [] #declare vectorlist for store the word2vec train representations
    validList = [] #declare vectorlist for store the word2vec validaiton representations

    def __init__(self, trainfilename, testfilename):
        df1=pd.read_csv(trainfilename)
        df2=pd.read_csv(testfilename)
        self.sent=df1['sentence'].values.tolist()
        self.end=df1['isEnding'].values.tolist()
        self.validSent=df2['sentence']
        self.validEnd=df2['isEnding']


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

    def train(self):
        for x in range(0,len(self.sent)):
            self.sentenceList.append(self.filter(self.sent[x]))
        for x in range(0,len(self.validSent)):
            self.sentenceList.append(self.filter(self.validSent[x]))

        
        self.model = Word2Vec(self.sentenceList, size=100, window=5, min_count=1, workers=4)


    def format(self, proba_test):
        probabs = []
        for x in range(0,len(proba_test)):
            probabs += str(proba_test[x]).split(" ")
        retProbabs = []
        for x in probabs:
            tmp = x.replace('[','')
            toAdd = tmp.replace(']','')
            retProbabs.append(toAdd)
        return retProbabs
   
        
        
    def determineSentenceVector(self, model_sentence):
        """
        This method takes a sentence and make a w2v representation from that.
        This will be an only one 100-dimensional vector
        """
        featureVec = np.zeros((100,), dtype="float32")
        nwords = 0

        for word in model_sentence:
            nwords = nwords+1
            featureVec = np.add(featureVec, self.model[word]) #kinyerjuk a modelbol a szohoz tartozo vectort es beletesszuk a featureVecbe

        if nwords>0:
            featureVec = np.divide(featureVec, nwords) #atlagoljuk a vektort a szavak szamaval
        return featureVec

    def getAllSentenceVector(self):
            
        #This method implements logistic regression for word2vec structures with given labels(1,0)
            

        for x in range(0,len(self.sent)):
            self.vectorList.append(self.determineSentenceVector(self.filter(self.sent[x])))

        for x in range(0,len(self.validSent)):
            self.validList.append(self.determineSentenceVector(self.filter(self.validSent[x])))
            
        X_train = scale(self.vectorList)
        LogReg = LogisticRegression(C=1, random_state=111)
        LogReg.fit(X_train, self.end)
        X_test = scale(self.validList)
        proba_test = LogReg.predict_proba(X_test)
        #print(proba_test)
        #a proba_test [0.2121 0.94242] parokat ad vissza minden validacios peldanyra

        probabs = self.format(proba_test) #kapunk egy listat az osszes valoszinusegrol
        # p,1-p parok is egymas mellett vannak, tehat kettesevel erdemes lepkedni
        print(len(probabs))
        

        count = 0
        falseNegativeCount = 0
        falsePositiveCount = 0
        i = 0
        j = 0
        while i < len(probabs):
            if probabs[i] < probabs[i+1] and self.validEnd[j] == 1:
                count+=1
            if probabs[i] > probabs[i+1] and self.validEnd[j] == 0:
                count+=1
            if probabs[i] < probabs[i+1] and self.validEnd[j] == 0:
                falseNegativeCount += 1
            if probabs[i] > probabs[i+1] and self.validEnd[j] == 1:
                falsePositiveCount += 1
            i += 2
            j += 1

        print("Accuracy: " + str(count/(len(probabs)/2)))
        print("False Positive: " + str(falsePositiveCount/(len(probabs)/2)))
        print("False Negative: " + str(falseNegativeCount/(len(probabs)/2)))



obj = convertToVec("end_or_not_train_data.csv", "val_isEnding.csv")
obj.train()
obj.getAllSentenceVector()
