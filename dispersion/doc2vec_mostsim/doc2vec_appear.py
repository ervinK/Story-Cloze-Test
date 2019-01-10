from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim import corpora, models, similarities
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from gensim.models import Word2Vec
import os
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import random
import numpy as np
import pandas as pd
from scipy import spatial
import sklearn
import csv

import nltk
nltk.download('punkt')



class Testing: #osztaly letrehozasa a feldolgozashoz
    filename = None
    model = None
    simCollectorNotEnding = {}
    simCollectorIsEnding = {}

    def __init__(self, model):
        self.model = model #model inicializalasa a tanitott modellel

    def getCosSums(self, list):
        sum = 0.0
        for x in list:
            sum += x[1]
        return sum


    def compare(self, filename):
        df=pd.read_csv(filename)
        self.s1=df['RandomFifthSentenceQuiz1'].values.tolist()
        self.s2=df['RandomFifthSentenceQuiz2'].values.tolist()
        self.decide=df['AnswerRightEnding'].values.tolist()
        with open('output_toPlot.csv', 'w', newline='') as f:
            fieldnames = ['id','appearance', 'isEnding']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for x in range(0,400):
                self.s1[x] = filterSentences(self.s1[x])
                s1_vector = model.infer_vector(self.s1[x])
                sims1 = self.model.docvecs.most_similar([s1_vector], topn=100)
                
            
                self.s2[x] = filterSentences(self.s2[x])
                s2_vector = model.infer_vector(self.s2[x])
                sims2 = self.model.docvecs.most_similar([s2_vector], topn=100)
                
                if self.decide == 1:
                    for y in range(0,len(sims2)):
                        try:
                            self.simCollectorNotEnding[sims2[y][0]] += 1
                        except:
                            self.simCollectorNotEnding[sims2[y][0]] = 0
                            self.simCollectorNotEnding[sims2[y][0]] += 1
                        try:
                            self.simCollectorIsEnding[sims1[y][0]] += 1
                        except:
                            self.simCollectorIsEnding[sims1[y][0]] = 0
                            self.simCollectorIsEnding[sims1[y][0]] += 1
                else:
                    for y in range(0,len(sims2)):
                        try:
                            self.simCollectorNotEnding[sims1[y][0]] += 1
                        except:
                            self.simCollectorNotEnding[sims1[y][0]] = 0
                            self.simCollectorNotEnding[sims1[y][0]] += 1
                        try:
                            self.simCollectorIsEnding[sims2[y][0]] += 1
                        except:
                            self.simCollectorIsEnding[sims2[y][0]] = 0
                            self.simCollectorIsEnding[sims2[y][0]] += 1
                print("The " + str(x) + ". round terminated")
            
            for key, value in self.simCollectorIsEnding.items():
                writer.writerow({'id' : key, 'appearance' : value, 'isEnding' : 1})
            for key, value in self.simCollectorNotEnding.items():
                writer.writerow({'id' : key, 'appearance' : value, 'isEnding' : 0})

        

def filterSentences(list): #train data filterezes
    string = list.lower()
    stopchars = ".,$?!;-:/%(){}[]^`\'\\\""
    stoppedstring = ""
    for x in string:
        if x not in stopchars:
            stoppedstring += x

    splittedstring = stoppedstring.split()
    stop_words = set(stopwords.words('english'))
    stop_words.add("would")
    stop_words.add("cannot")
    stop_words.add("should")
    finalstring = []
    for x in splittedstring:
        if x not in stop_words:
            finalstring.append(x)
    return finalstring



def filterData(list): #train data filterezes
    filteredList = []
    stopchars = ".,$?!;-:/%(){}[]^`\'\\\""
    stop_words = set(stopwords.words('english'))
    stop_words.add("would")
    stop_words.add("cannot")
    stop_words.add("should")

    for x in list:
        processedStr=x
        toFinalStr = ""
        if "\'s" in x:
            processedStr = processedStr.replace("\'s","")
        if "\'ll" in x:
            processedStr = processedStr.replace("\'ll","")
        if "n\'t" in x:
            processedStr = processedStr.replace("n\'t","")
        if "\'d" in x:
            processedStr = processedStr.replace("\'d","")
        if "\'ve" in x:
            processedStr = processedStr.replace("\'ve","")
        if "\'m" in x:
            processedStr = processedStr.replace("\'m","")
        if "o\'" in x:
            processedStr = processedStr.replace("o\'","")
        for y in processedStr:
            if y not in stopchars:
                toFinalStr += y.lower()
        toFinalStr = toFinalStr.split()
        finalStr = ""

        for iter in toFinalStr:
            if iter not in stop_words:
                finalStr += (str(iter) + " ")
        #print(finalStr)
        if(len(finalStr)!=0):
            filteredList.append(finalStr)

    return filteredList

def readTrainingData(filename): #fuggveny trainDataBeolvasasara
    df=pd.read_csv(filename) #open file
    s1=df['sentence5'].values.tolist() #read columns
    s1 = filterData(s1)
    c = s1 # a beolvasott oszlopokat konkatenaljuk a corpusba
    return c

c1 = readTrainingData("rcs_train.csv")

corpus = c1

data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
#print(data)

model = Doc2Vec(data, alpha=0.025, min_alpha=0.025, vector_size=100, window=2, min_count=1, workers=4)

t1 = Testing(model)
t1.compare("rcs_valid.csv")
