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
import csv


class convertToWmdForm:
    filename = None
    model = None
    resultFileName = None

    def __init__(self, filename, resultFileName):
        df=pd.read_csv(filename)
        self.s1=df['InputSentence1'].values.tolist()
        self.s2=df['InputSentence2'].values.tolist()
        self.s3=df['InputSentence3'].values.tolist()
        self.s4=df['InputSentence4'].values.tolist()
        self.r1=df['RandomFifthSentenceQuiz1'].values.tolist()
        self.r2=df['RandomFifthSentenceQuiz2'].values.tolist()
        self.d=df['AnswerRightEnding'].values.tolist()
        self.resultFileName = resultFileName

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

    def trainFilteredSentences(self):
        for x in range(0,len(self.s1)):
            self.r1[x] = self.filter(self.r1[x]) #filter the sentences1,2,3,4, Right_ending and the Wrong_ending
            self.r2[x] = self.filter(self.r2[x])
            self.s1[x] = self.filter(self.s1[x])
            self.s2[x] = self.filter(self.s2[x])
            self.s3[x] = self.filter(self.s3[x])
            self.s4[x] = self.filter(self.s4[x])


        vocab = self.r1 + self.r2 + self.s1 + self.s2 + self.s3 + self.s4
        self.model = Word2Vec(vocab, size=100, window=5, min_count=1, workers=4)

        with open(self.resultFileName, 'w', newline='') as f:
            fieldnames = ['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4', 'AnswerRightEnding', 'sumOfWmd']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for x in range(0,len(self.s1)):
                if len(self.s1[x]) == 0 or len(self.s2[x]) == 0 or len(self.s3[x]) == 0 or len(self.s4[x]) == 0 or len(self.r1[x]) == 0 or len(self.r2[x]) == 0:
                    continue
                if self.d[x] == 1:
                    writer.writerow({'InputSentence1' : str(self.model.wmdistance(self.s1[x], self.r1[x])), 'InputSentence2' : str(self.model.wmdistance(self.s2[x], self.r1[x])), 'InputSentence3' : str(self.model.wmdistance(self.s3[x], self.r1[x])), 'InputSentence4' : str(self.model.wmdistance(self.s4[x], self.r1[x])), 'AnswerRightEnding' : '1', 'sumOfWmd' : str(self.model.wmdistance(self.s1[x], self.r1[x]) + self.model.wmdistance(self.s2[x], self.r1[x]) + self.model.wmdistance(self.s3[x], self.r1[x]) + self.model.wmdistance(self.s4[x], self.r1[x]))})
                    writer.writerow({'InputSentence1' : str(self.model.wmdistance(self.s1[x], self.r2[x])), 'InputSentence2' : str(self.model.wmdistance(self.s2[x], self.r2[x])), 'InputSentence3' : str(self.model.wmdistance(self.s3[x], self.r2[x])), 'InputSentence4' : str(self.model.wmdistance(self.s4[x], self.r2[x])), 'AnswerRightEnding' : '0', 'sumOfWmd' : str(self.model.wmdistance(self.s1[x], self.r2[x]) + self.model.wmdistance(self.s2[x], self.r2[x]) + self.model.wmdistance(self.s3[x], self.r2[x]) + self.model.wmdistance(self.s4[x], self.r2[x]))})

                else:
                    writer.writerow({'InputSentence1' : str(self.model.wmdistance(self.s1[x], self.r1[x])), 'InputSentence2' : str(self.model.wmdistance(self.s2[x], self.r1[x])), 'InputSentence3' : str(self.model.wmdistance(self.s3[x], self.r1[x])), 'InputSentence4' : str(self.model.wmdistance(self.s4[x], self.r1[x])), 'AnswerRightEnding' : '0', 'sumOfWmd' : str(self.model.wmdistance(self.s1[x], self.r1[x]) + self.model.wmdistance(self.s2[x], self.r1[x]) + self.model.wmdistance(self.s3[x], self.r1[x]) + self.model.wmdistance(self.s4[x], self.r1[x]))})
                    writer.writerow({'InputSentence1' : str(self.model.wmdistance(self.s1[x], self.r2[x])), 'InputSentence2' : str(self.model.wmdistance(self.s2[x], self.r2[x])), 'InputSentence3' : str(self.model.wmdistance(self.s3[x], self.r2[x])), 'InputSentence4' : str(self.model.wmdistance(self.s4[x], self.r2[x])), 'AnswerRightEnding' : '1', 'sumOfWmd' : str(self.model.wmdistance(self.s1[x], self.r2[x]) + self.model.wmdistance(self.s2[x], self.r2[x]) + self.model.wmdistance(self.s3[x], self.r2[x]) + self.model.wmdistance(self.s4[x], self.r2[x]))})


traindataProcess = convertToWmdForm("cloze_valid.csv","train_wmd.csv")
traindataProcess.trainFilteredSentences()
validationProcess = convertToWmdForm("cloze_test.csv","test_wmd.csv")
validationProcess.trainFilteredSentences()
