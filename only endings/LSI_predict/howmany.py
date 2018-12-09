import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models, similarities
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from gensim.models import Word2Vec
import os
import nltk
from nltk.corpus import stopwords
import random
import numpy as np
import pandas as pd
import csv
from collections import defaultdict

class LsiSimsCalc:
    testfile = None
    valfile = None
    lsi = None
    dictionary = None
    corpus = None
    simCollector = {}

    def __init__(self, testfile, valfile):
        df1 = pd.read_csv(testfile)
        df2 = pd.read_csv(valfile)
        self.s5 = df1['sentence5'].values.tolist() #beolvassuk az 5. mondatokat a train databol
        self.end1 = df2['RandomFifthSentenceQuiz1'] #beolvassuk a befejezest a validacios halmazbol
        self.end2 = df2['RandomFifthSentenceQuiz2'] #beolvassuk a befejezest a validacios halmazbol
        self.dir = df2['AnswerRightEnding'] #beolvassuk a helyes befejezes indexet

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

    def filterTexts(self):
        for i in range(0, len(self.s5)):
            self.s5[i] = self.filter(self.s5[i]) #remove stopwords from the train sentences
        for i in range(0, len(self.end1)):
            self.end1[i] = self.filter(self.end1[i]) #remove stopwords from the validation sentences
            self.end2[i] = self.filter(self.end2[i]) #remove stopwords from the validation sentences
        print("Traintext filtered")
        return self.s5

    def processString(self, pair):
        for x in pair:
            return x

    def createLsi(self):
        frequency = defaultdict(int)
        for text in self.s5:
             for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 1]
                 for text in self.s5]
        #print(texts)
        self.dictionary = corpora.Dictionary(texts) #initialize dictionary
        self.corpus=[self.dictionary.doc2bow(text) for text in texts] #BAG OF WORDS
        self.lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=30) #initialize lsi model

    def determineSims(self, topDepth):
        right_acc = 0
        wrong_acc = 0
        for i in range(0, len(self.end1)):
           
            #print(self.end2[i])
            print("Determining in process")
            vec_sentence1 = self.dictionary.doc2bow(self.end1[i])
            vec_lsi1 = self.lsi[vec_sentence1]
            index1 = similarities.MatrixSimilarity(self.lsi[self.corpus])
            sims1 = index1[vec_lsi1]
            simlist1 = list(enumerate(sims1))

            print("A vizsgalt mondat: " + str(self.end1[i]))
            simlist1.sort(key=lambda x: x[1],reverse=True)
            
            iter = 0
            while iter < topDepth:
                try:
                    self.simCollector[self.processString(simlist1[iter])] += 1
                except:
                    self.simCollector[self.processString(simlist1[iter])] = 0
                    self.simCollector[self.processString(simlist1[iter])] += 1
                iter += 1
        for x in self.simCollector:
            print(str(x) + " " + str(self.simCollector[x]))
             

obj = LsiSimsCalc("rcs_train17.csv","rcs_val16.csv")
obj.filterTexts()
obj.createLsi()
obj.determineSims(70)