import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim import corpora, models, similarities
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from gensim.models import Word2Vec
import os
import nltk
from nltk.corpus import stopwords
import random
import numpy as np
import pandas as pd

class Testing:
    filename = None
    model = None
    def __init__(self, filename, model):
        df=pd.read_csv(filename)
        self.s1=df['InputSentence1'].values.tolist()
        self.s2=df['InputSentence2'].values.tolist()
        self.s3=df['InputSentence3'].values.tolist()
        self.s4=df['InputSentence4'].values.tolist()
        self.r1=df['RandomFifthSentenceQuiz1'].values.tolist()
        self.r2=df['RandomFifthSentenceQuiz2'].values.tolist()
        self.d=df['AnswerRightEnding'].values.tolist()
        self.model = model

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

    def filter(self,sentence):
        stop_words = set(stopwords.words('english')) #nltk stopwords
        stop_words.add("would")
        stop_words.add("cannot")
        sentence = sentence.lower().split() #lowercase all of the words in the dataset
        sentence = self.deleteStopChars(sentence) #filtering to stopchars
        sentence = [w for w in sentence if w not in stop_words] #filtering to stopwords
        return sentence #return with a filtered sentence

    def calculateAccuracy(self):

        rightCount = 0.0 #count the positive results
        wrongCount = 0.0 #count the wrong results
        skipCounter = 0
        sentenceSkip = 0
        equalityCounter = 0 #count equal values of WMD


        skipValue = 0

        for x in range(0,len(self.s1)):

            lst1 = []
            lst2 = []
            self.r1[x] = self.filter(self.r1[x]) #filter the sentences1,2,3,4, Right_ending and the Wrong_ending
            self.r2[x] = self.filter(self.r2[x])
            self.s1[x] = self.filter(self.s1[x])
            self.s2[x] = self.filter(self.s2[x])
            self.s3[x] = self.filter(self.s3[x])
            self.s4[x] = self.filter(self.s4[x])


            d1 = 0
            d2 = 0
            d3 = 0
            d4 = 0
            d5 = 0
            d6 = 0
            d7 = 0
            d8 = 0


            for y in self.r1[x]:
                for z in self.s1[x]:
                    d1 += model.wv.similarity(y,z)

                for z in self.s2[x]:
                    d2 += model.wv.similarity(y,z)

                for z in self.s3[x]:
                    d3 += model.wv.similarity(y,z)

                for z in self.s4[x]:
                    d4 += model.wv.similarity(y,z)


            if len(self.r1[x]) == 0  or len(self.s1[x]) == 0 or len(self.s2[x]) == 0 or len(self.s3[x]) == 0 or len(self.s4[x]) == 0:
                sentenceSkip += 1
                continue
            else:
                d1 = d1 /(len(self.r1[x])*len(self.s1[x]))
                lst1.append(d1)
                d2 = d2 /(len(self.r1[x])*len(self.s2[x]))
                lst1.append(d2)
                d3 = d3 /(len(self.r1[x])*len(self.s3[x]))
                lst1.append(d3)
                d4 = d4 /(len(self.r1[x])*len(self.s4[x]))
                lst1.append(d4)



            for y in self.r2[x]:
                for z in self.s1[x]:
                    d5 += model.wv.similarity(y,z)
                for z in self.s2[x]:
                    d6 += model.wv.similarity(y,z)

                for z in self.s3[x]:
                    d7 += model.wv.similarity(y,z)

                for z in self.s4[x]:
                    d8 += model.wv.similarity(y,z)

            if len(self.r2[x]) == 0  or len(self.s1[x]) == 0 or len(self.s2[x]) == 0 or len(self.s3[x]) == 0 or len(self.s4[x]) == 0:
                sentenceSkip += 1
                continue
            else:
                d5 = d5 /(len(self.r2[x])*len(self.s1[x]))
                lst2.append(d5)
                d6 = d6 /(len(self.r2[x])*len(self.s2[x]))
                lst2.append(d6)
                d7 = d7 /(len(self.r2[x])*len(self.s3[x]))
                lst2.append(d7)
                d8 = d8 /(len(self.r2[x])*len(self.s4[x]))
                lst2.append(d8)




            dec1 = max(lst1)-min(lst1)
            dec2 = max(lst2)-min(lst2)
            if self.d[x] == 1 and dec1 > dec2:
                rightCount += 1
            elif self.d[x] == 2 and dec2 > dec1:
                rightCount += 1
            else:
                wrongCount += 1


            if dec1 == dec2:
                equalityCounter += 1

        print("Right answer accurancy: " + str(rightCount/(len(self.s1)-sentenceSkip))) #the number of right answers of the agent
        print("Wrong answer accurancy: " + str(wrongCount/(len(self.s1)-sentenceSkip))) #the number of false-positive answers of the agent
        print("Number of COS equalities: " + str(equalityCounter))
        print("Skipped words: " + str(skipCounter))
        print("Skipped sentences: " + str(sentenceSkip))

def readTrainingData(filename):
    df=pd.read_csv(filename) #open file
    s1=df['sentence1'].values.tolist() #read columns
    s2=df['sentence2'].values.tolist()
    s3=df['sentence3'].values.tolist()
    s4=df['sentence4'].values.tolist()
    c = s1 + s2 + s3 + s4 #concatenate the columns of training data for the corpus
    return c


def readTrainingDataB(filename):
    df=pd.read_csv(filename) #open file
    s1=df['InputSentence1'].values.tolist() #read columns
    s2=df['InputSentence2'].values.tolist()
    s3=df['InputSentence3'].values.tolist()
    s4=df['InputSentence4'].values.tolist()
    r1=df['RandomFifthSentenceQuiz1'].values.tolist()
    r2=df['RandomFifthSentenceQuiz2'].values.tolist()
    c = s1 + s2 + s3 + s4 + r1 + r2 #concatenate the columns of training data for the corpus
    return c
def deleteStopCharsCorpus(str): #delete stop chars from test cases
    stopchars = ".,$?!€;-:/%(){}[]^`\'\"\\" #stop chars
    newstr = ""
    for x in range(0,len(str)): #iterating through a sentence
        if str[x] not in stopchars: #if a char not in the stoplist it can go forward
                newstr += str[x] #adding to the filtered word

    return newstr #return with the sentence

def stopFilter(corpus):
    tok_corp = [nltk.word_tokenize(sent) for sent in corpus] #first make tokens from corpus
    stop_words = set(stopwords.words('english')) #load stopwords
    stop_words.add("n\'t")
    stop_words.add("\'s")
    stop_words.add("\'ll")
    stop_words.add("\'d")
    stop_words.add("\'m")
    stop_words.add("\'ve")
    stop_words.add("\'")
    stop_words.add("will")
    stop_words.add("would")
    tok_again = [] #this is going to be the return list

    for x in range(0,len(tok_corp)): #iterate through sentences

        tmp = [] #collector for the words of a sentence
        l = len(tok_corp[x])
        for y in range(0,l): #iterate through words
            if tok_corp[x][y].lower() not in stop_words: #if it is not in the stoplist
                lowWord = tok_corp[x][y].lower()
                word = deleteStopCharsCorpus(lowWord)
                tmp.append(word) #add to the temporary word collector
            else:
                continue
        tok_again.append(tmp) #add a sentence to the sentence list
    return tok_again #return with the list of filtered sentences


def concatenateCorpuses(c1, c2):
    list = []
    for x in c1:
        list.append(x)
    for x in c2:
        list.append(x)
    return list


c1 = readTrainingData("test_train.csv") #read file
c1 = stopFilter(c1) #filtering the read data
c2 = readTrainingDataB("rcs_test.csv")
c2 = stopFilter(c2)

corpus = concatenateCorpuses(c1,c2)
model = gensim.models.Word2Vec(corpus, size=100, window=5, min_count=1)

t1 = Testing("rcs_test.csv",model)
t1.calculateAccuracy()
#print(model.wv.similarity("young","ambivalence"))
"""print(c2)

#szuvasodas, baromsag

corpus = concatenateCorpuses(c1,c2)

model = gensim.models.Word2Vec(corpus, size=100, window=5, min_count=2)

print(model.wv.similarity(v1 = "disbelief",v2 = "szuvasodas"))"""
