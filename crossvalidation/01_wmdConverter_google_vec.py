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

    def __init__(self, filename1, filename2):
        df1=pd.read_csv(filename1)
        df2=pd.read_csv(filename2)
        self.s1=df1['InputSentence1'].values.tolist()
        self.s2=df1['InputSentence2'].values.tolist()
        self.s3=df1['InputSentence3'].values.tolist()
        self.s4=df1['InputSentence4'].values.tolist()
        self.r1=df1['RandomFifthSentenceQuiz1'].values.tolist()
        self.r2=df1['RandomFifthSentenceQuiz2'].values.tolist()
        self.d=df1['AnswerRightEnding'].values.tolist()
        self.ss1=df2['InputSentence1'].values.tolist()
        self.ss2=df2['InputSentence2'].values.tolist()
        self.ss3=df2['InputSentence3'].values.tolist()
        self.ss4=df2['InputSentence4'].values.tolist()
        self.rr1=df2['RandomFifthSentenceQuiz1'].values.tolist()
        self.rr2=df2['RandomFifthSentenceQuiz2'].values.tolist()
        self.dd=df2['AnswerRightEnding'].values.tolist()

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

    def trainFilteredSentences(self, outFileName1,outFileName2):
        for x in range(0,len(self.s1)):
            self.r1[x] = self.filter(self.r1[x]) #filter the sentences1,2,3,4, Right_ending and the Wrong_ending
            self.r2[x] = self.filter(self.r2[x])
            self.s1[x] = self.filter(self.s1[x])
            self.s2[x] = self.filter(self.s2[x])
            self.s3[x] = self.filter(self.s3[x])
            self.s4[x] = self.filter(self.s4[x])
        for x in range(0,len(self.ss1)):
            self.rr1[x] = self.filter(self.rr1[x]) #filter the sentences1,2,3,4, Right_ending and the Wrong_ending
            self.rr2[x] = self.filter(self.rr2[x])
            self.ss1[x] = self.filter(self.ss1[x])
            self.ss2[x] = self.filter(self.ss2[x])
            self.ss3[x] = self.filter(self.ss3[x])
            self.ss4[x] = self.filter(self.ss4[x])


        vocab = self.r1 + self.r2 + self.s1 + self.s2 + self.s3 + self.s4
        vocab2 = self.rr1 + self.rr2 + self.ss1 + self.ss2 + self.ss3 + self.ss4
        self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

        with open(outFileName1, 'w', newline='') as f:
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
        with open(outFileName2, 'w', newline='') as f:
            fieldnames = ['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4', 'AnswerRightEnding', 'sumOfWmd']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for x in range(0,len(self.ss1)):
                if len(self.ss1[x]) == 0 or len(self.ss2[x]) == 0 or len(self.ss3[x]) == 0 or len(self.ss4[x]) == 0 or len(self.rr1[x]) == 0 or len(self.rr2[x]) == 0:
                    continue
                if self.dd[x] == 1:
                    writer.writerow({'InputSentence1' : str(self.model.wmdistance(self.ss1[x], self.rr1[x])), 'InputSentence2' : str(self.model.wmdistance(self.ss2[x], self.rr1[x])), 'InputSentence3' : str(self.model.wmdistance(self.ss3[x], self.rr1[x])), 'InputSentence4' : str(self.model.wmdistance(self.ss4[x], self.rr1[x])), 'AnswerRightEnding' : '1', 'sumOfWmd' : str(self.model.wmdistance(self.ss1[x], self.rr1[x]) + self.model.wmdistance(self.ss2[x], self.rr1[x]) + self.model.wmdistance(self.ss3[x], self.rr1[x]) + self.model.wmdistance(self.ss4[x], self.rr1[x]))})
                    writer.writerow({'InputSentence1' : str(self.model.wmdistance(self.ss1[x], self.rr2[x])), 'InputSentence2' : str(self.model.wmdistance(self.ss2[x], self.rr2[x])), 'InputSentence3' : str(self.model.wmdistance(self.ss3[x], self.rr2[x])), 'InputSentence4' : str(self.model.wmdistance(self.ss4[x], self.rr2[x])), 'AnswerRightEnding' : '0', 'sumOfWmd' : str(self.model.wmdistance(self.ss1[x], self.rr2[x]) + self.model.wmdistance(self.ss2[x], self.rr2[x]) + self.model.wmdistance(self.ss3[x], self.rr2[x]) + self.model.wmdistance(self.ss4[x], self.rr2[x]))})

                else:
                    writer.writerow({'InputSentence1' : str(self.model.wmdistance(self.ss1[x], self.rr1[x])), 'InputSentence2' : str(self.model.wmdistance(self.ss2[x], self.rr1[x])), 'InputSentence3' : str(self.model.wmdistance(self.ss3[x], self.rr1[x])), 'InputSentence4' : str(self.model.wmdistance(self.ss4[x], self.rr1[x])), 'AnswerRightEnding' : '0', 'sumOfWmd' : str(self.model.wmdistance(self.ss1[x], self.rr1[x]) + self.model.wmdistance(self.ss2[x], self.rr1[x]) + self.model.wmdistance(self.ss3[x], self.rr1[x]) + self.model.wmdistance(self.ss4[x], self.rr1[x]))})
                    writer.writerow({'InputSentence1' : str(self.model.wmdistance(self.ss1[x], self.rr2[x])), 'InputSentence2' : str(self.model.wmdistance(self.ss2[x], self.rr2[x])), 'InputSentence3' : str(self.model.wmdistance(self.ss3[x], self.rr2[x])), 'InputSentence4' : str(self.model.wmdistance(self.ss4[x], self.rr2[x])), 'AnswerRightEnding' : '1', 'sumOfWmd' : str(self.model.wmdistance(self.ss1[x], self.rr2[x]) + self.model.wmdistance(self.ss2[x], self.rr2[x]) + self.model.wmdistance(self.ss3[x], self.rr2[x]) + self.model.wmdistance(self.ss4[x], self.rr2[x]))})


traindataProcess = convertToWmdForm("cloze_valid.csv","cloze_test.csv")
traindataProcess.trainFilteredSentences("train_wmd.csv","test_wmd.csv")
