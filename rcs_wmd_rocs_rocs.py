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
from timeit import default_timer as timer
from numba import vectorize
from numba import cuda




def readTrainingData(filename):
    df=pd.read_csv(filename) #open file
    s1=df['sentence1'].values.tolist() #read columns
    s2=df['sentence2'].values.tolist()
    s3=df['sentence3'].values.tolist()
    s4=df['sentence4'].values.tolist()
    c = s1 + s2 + s3 + s4 #concatenate the columns of training data for the corpus
    return c

def stopFilter(corpus):
    tok_corp = [nltk.word_tokenize(sent) for sent in corpus] #first make tokens from corpus
    stop_words = set(stopwords.words('english')) #load stopwords
    stopchars = ".,$?!€;-:/%(){}[]" #potential stopcharacters
    for x in stopchars: #iterate through
        stop_words.add(x)
    tok_again = [] #this is going to be the return list

    for x in range(0,len(tok_corp)): #iterate through sentences
        tmp = [] #collector for the words of a sentence
        l = len(tok_corp[x])
        for y in range(0,l): #iterate through words
            if tok_corp[x][y] not in stop_words: #if it is not in the stoplist
                tok_corp[x][y] = tok_corp[x][y].lower()
                tmp.append(tok_corp[x][y]) #add to the temporary word collector
        tok_again.append(tmp) #add a sentence to the sentence list
    return tok_again #return with the list of filtered sentences

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
        stopchars = ".,$?!€;-:/%(){}[]" #stop chars
        newlist = [] #list of the filtered sentence
        for x in range(0,len(str)): #iterating through a sentence
            willFlag = 0
            notFlag = 0
            newstr = ""
            apostr = 0
            processedStr=""
            if "'s" in str[x]:
                processedStr = str[x].replace("\'s","")
            elif "'ll" in str[x]:
                processedStr = str[x].replace("\'ll","")
                willFlag = 1
            elif "'n't" in str[x]:
                processedStr = str[x].replace("n\'t","")
                notFlag = 1

            else:
                processedStr = str[x]
            for y in range(0,len(processedStr)): #iterating through a word
                if processedStr[y] not in stopchars: #if a char not in the stoplist it can go forward
                    newstr += processedStr[y] #adding to the filtered word
            newlist.append(newstr) #add the filtered word to a sentence
            if willFlag == 1:
                willFlag = 0
                newlist.append("will")
            if notFlag == 1:
                notFlag = 0
                newlist.append("not")
        return newlist #return with the sentence

    def filter(self,sentence):
        stop_words = set(stopwords.words('english')) #nltk stopwords
        sentence = sentence.lower().split() #lowercase all of the words in the dataset
        sentence = self.deleteStopChars(sentence) #filtering to stopchars
        sentence = [w for w in sentence if w not in stop_words] #filtering to stopwords
        return sentence #return with a filtered sentence

    def calculateAccuracy(self):

        rightCount = 0.0 #count the positive results
        wrongCount = 0.0 #count the false-positive results

        equalityCounter = 0 #count equal values of WMD

        varInRight = 0.0 #we are going to calculate variance for min WMD sentences in right cases
        varInWrong = 0.0 #we are going to calculate variance for min WMD sentences in false-positive cases
        skipValue = 0
        """WMD can not processing one word against a sentence.
        Because the filtering it could happen at the end of stoplisting we will get a one-word sentence which is uncalculatable """
        for x in range(0,len(self.s1)):
            if(len(self.r1[x]) == 1 or len(self.r2[x]) == 1 or len(self.s1[x]) == 1 or len(self.s2[x]) == 1 or len(self.s3[x]) == 1 or len(self.s4[x]) == 1): #if the previous happens we skip this iteration
                skipValue += 1
                continue
            lst1 = []
            self.r1[x] = self.filter(self.r1[x]) #filter the sentences1,2,3,4, Right_ending and the Wrong_ending
            self.r2[x] = self.filter(self.r2[x])
            self.s1[x] = self.filter(self.s1[x])
            self.s2[x] = self.filter(self.s2[x])
            self.s3[x] = self.filter(self.s3[x])
            self.s4[x] = self.filter(self.s4[x])
            d1 = self.model.wmdistance(self.s1[x], self.r1[x]) #calculate WMD for 1st ending
            d2 = self.model.wmdistance(self.s2[x], self.r1[x])
            d3 = self.model.wmdistance(self.s3[x], self.r1[x])
            d4 = self.model.wmdistance(self.s4[x], self.r1[x])


            lst1.append(d1) #appending WMD of sentences to a list
            lst1.append(d2)
            lst1.append(d3)
            lst1.append(d4)

            dec1 = max(lst1)-min(lst1) # max-min test
            #dec1 = min(lst1) # min-test
            #dec1 = sum(lst1)/len(lst1) #avg-test
            #dec1 = sum(lst1) #sum-test


            vari1 = abs((sum(lst1)/len(lst1))-min(lst1)) #sub-calculate for the variance

            lst2 = []
            d5 = self.model.wmdistance(self.s1[x], self.r2[x]) #calculate WMD for 2nd ending
            d6 = self.model.wmdistance(self.s2[x], self.r2[x])
            d7 = self.model.wmdistance(self.s3[x], self.r2[x])
            d8 = self.model.wmdistance(self.s4[x], self.r2[x])

            lst2.append(d5) #appending WMD of endings to a list
            lst2.append(d6)
            lst2.append(d7)
            lst2.append(d8)

            dec2 = max(lst2)-min(lst2) # max-min test
            #dec2 = min(lst2) #min-test
            #dec2 = sum(lst2)/len(lst2)  #avg-test
            #dec2 = sum(lst2) #sum-test

            dec2 = max(lst2)

            vari2 = abs((sum(lst2)/len(lst2))-min(lst2)) #sub-calculate for the variance

            if self.d[x] == 1 and dec1 < dec2: # If the 1st ending has lower words moving distance
                rightCount += 1 #count it
                varInRight += vari1 #adding to variance variable

            elif self.d[x] == 2 and dec2 < dec1: # If the 2nd ending has lower words moving distance
                rightCount += 1 #count it
                varInRight += vari2 #adding to variance variable

            elif dec2 == dec1: # If the 2 endings have equal words moving distance
                equalityCounter += 1 #count it
                lst1.remove(min(lst1)) #then remove the minimum WMD sentence from the pool
                lst2.remove(min(lst2)) #then remove the minimum WMD sentence from the other pool
                if max(lst1)-min(lst1) > max(lst2)-min(lst2) and self.d[x] == 2: #calculate max-min again

                    rightCount += 1
                elif max(lst1)-min(lst1) < max(lst2)-min(lst2) and self.d[x] == 1: #calculate max-min again

                    rightCount += 1
                elif max(lst1)-min(lst1) > max(lst2)-min(lst2) and self.d[x] == 1: #calculate max-min again
                    wrongCount += 1

                elif max(lst1)-min(lst1) < max(lst2)-min(lst2) and self.d[x] == 2: #calculate max-min again
                    wrongCount += 1

                elif max(lst1)-min(lst1) == max(lst2)-min(lst2): #if they are still equal pick a random ending
                    ran = random.randint(1,2)
                    if ran == 2 and self.d[x] == 2:
                        rightCount += 1

                    elif ran == 1 and self.d[x] == 1:
                        rightCount += 1

                    elif ran == 2 and self.d[x] == 1:
                        wrongCount += 1

                    elif ran == 1 and self.d[x] == 2:
                        wrongCount += 1



            elif self.d[x] == 2 and dec1 < dec2: #false-positive case
                wrongCount += 1 #count it
                varInWrong += vari1

            elif self.d[x] == 1 and dec1 > dec2: #false-positive case
                wrongCount += 1 #count it
                varInWrong += vari2

        #if we skipped an iteration skipValue marks it
        print("Right answer accurancy: " + str(rightCount/len(self.s1)-skipValue)) #the number of right answers of the agent
        print("Wrong answer accurancy: " + str(wrongCount/len(self.s1)-skipValue)) #the number of false-positive answers of the agent

        print("Variance of min value in right cases: " + str(varInRight/rightCount)) #difference of the min WMD sentece from the avg WMD sentence in positive cases
        print("Variance of min value in wrong cases: " + str(varInWrong/wrongCount)) #difference of the min WMD sentece from the avg WMD sentence in false-positive cases
        print("Number of WMD equalities: " + str(equalityCounter))



model = Word2Vec.load("word2vec.model") #load model


start = timer()
t1 = Testing("rcs_test.csv",model)
t1.calculateAccuracy()
vectoradd_time = timer() - start
print(str(vectoradd_time) + " ms")
