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


class Testing: #osztaly letrehozasa a feldolgozashoz
    filename = None
    model = None
    def __init__(self, filename, model):
        df=pd.read_csv(filename)
        self.s1=df['InputSentence1'].values.tolist() #teszt adatok oszlopainak beolvasasa
        self.s2=df['InputSentence2'].values.tolist() #teszt adatok oszlopainak beolvasasa
        self.s3=df['InputSentence3'].values.tolist() #teszt adatok oszlopainak beolvasasa
        self.s4=df['InputSentence4'].values.tolist() #teszt adatok oszlopainak beolvasasa
        self.r1=df['RandomFifthSentenceQuiz1'].values.tolist() #teszt adatok oszlopainak beolvasasa
        self.r2=df['RandomFifthSentenceQuiz2'].values.tolist() #teszt adatok oszlopainak beolvasasa
        self.d=df['AnswerRightEnding'].values.tolist()
        self.model = model #model inicializalasa a tanitott modellel

    def stringFilter(self, x): #fuggveny a beolvasott adatok elokeszitesere
        stop_words = set(stopwords.words('english')) #nltk.corpus beepitett stopwordok
        stop_words.add("would") #nem volt benne, hozzaadjuk
        stop_words.add("cannot") #nem volt benne, hozzaadjuk
        stopchars = ".,$?!€;-:/%(){}[]^`\\\'\"" #specialis karakterek
        processedStr = x

        if "\'s" in x:
            processedStr = processedStr.replace("\'s","") #contractionok kiszurese
        if "\'ll" in x:
            processedStr = processedStr.replace("\'ll","") #contractionok kiszurese
        if "n\'t" in x:
            processedStr = processedStr.replace("n\'t","") #contractionok kiszurese
        if "\'d" in x:
            processedStr = processedStr.replace("\'d","") #contractionok kiszurese
        if "\'ve" in x:
            processedStr = processedStr.replace("\'ve","") #contractionok kiszurese
        if "\'m" in x:
            processedStr = processedStr.replace("\'m","") #contractionok kiszurese
        if "o\'" in x:
            processedStr = processedStr.replace("o\'","") #contractionok kiszurese

        toFinalStr = ""
        for y in processedStr:
            if y not in stopchars: #specialis karakterek kiszurese
                toFinalStr += y.lower()
        toFinalStr = toFinalStr.split() #listava alakitas
        finalStr = ""

        for iter in toFinalStr: #stopwordok kiszurese listabol
            if iter not in stop_words:
                finalStr += (str(iter) + " ") #visszaalakitas stringge
        return finalStr


    def calculateAccuracy(self): #fuggveny a filterezes megvalositasara es a cosinus similarity modszer pontossaganak meresere

        rightCount = 0.0 #helyes megallapitasok kiszurese
        wrongCount = 0.0 #helytelen megallapitasok kiszurese
        equalityCounter = 0 #egyenloseg esetek

        for x in range(0,len(self.s1)): #vegigiteralunk az allitasokon es a kovetkezmenyeken
            lst = []
            dec1 = 0.0 #ide gyujtjuk az elso kovetkezmenyre mert hasonlosagat a mondatoknak
            dec2 = 0.0 #ide gyujtjuk a masodik kovetkezmenyre mert hasonlosagat a mondatoknak
            self.s1[x] = self.stringFilter(self.s1[x]) #mondatok filterezese
            self.s2[x] = self.stringFilter(self.s2[x]) #mondatok filterezese
            self.s3[x] = self.stringFilter(self.s3[x]) #mondatok filterezese
            self.s4[x] = self.stringFilter(self.s4[x]) #mondatok filterezese
            self.r1[x] = self.stringFilter(self.r1[x]) #mondatok filterezese
            self.r2[x] = self.stringFilter(self.r2[x]) #mondatok filterezese

            r1_vec = avg_feature_vector(self.r1[x].split(), 100, self.model) #vegzodesek szavainak osszegzese vektorra
            r2_vec = avg_feature_vector(self.r2[x].split(), 100, self.model) #vegzodesek szavainak osszegzese vektorra

            s1_vec = avg_feature_vector(self.s1[x].split(), 100, self.model) #mondatok szavainak osszegzese vektorra
            s2_vec = avg_feature_vector(self.s2[x].split(), 100, self.model) #mondatok szavainak osszegzese vektorra
            s3_vec = avg_feature_vector(self.s3[x].split(), 100, self.model) #mondatok szavainak osszegzese vektorra
            s4_vec = avg_feature_vector(self.s4[x].split(), 100, self.model) #mondatok szavainak osszegzese vektorra
            lst.append(s1_vec) #vektorok listaba fuzese
            lst.append(s2_vec) #vektorok listaba fuzese
            lst.append(s3_vec) #vektorok listaba fuzese
            lst.append(s4_vec) #vektorok listaba fuzese
            for m in lst: #vegigiteralunk a vektorlistan
                res1 = 1.0 - spatial.distance.cosine(m, r1_vec) #cos sim szamolasa elso vegzodesre
                res2 = 1.0 - spatial.distance.cosine(m, r2_vec) #cos sim szamolasa masodik vegzodesre

                dec1 += res1 #elso vegzodeshez kapcsolodo mondatok hasonlosaganak osszegei
                dec2 += res2 #masodik vegzodeshez kapcsolodo mondatok hasonlosaganak osszegei

            if self.d[x] == 1 and dec1 > dec2: #ha az elso vegzodeshez nagyobb a hasonlosag es talalat van
                rightCount += 1
            elif self.d[x] == 2 and dec2 > dec1: #ha a masodik vegzodeshez nagyobb a hasonlosag es talalat van
                rightCount += 1
            elif dec1 == dec2: #egyenlosegek szamlalasa
                equalityCounter += 1
                continue
            else:
                wrongCount += 1
        print("Right answer accurancy: " + str(rightCount/(len(self.s1)-equalityCounter))) #helyes valaszok szazalekosan
        print("Wrong answer accurancy: " + str(wrongCount/(len(self.s1)-equalityCounter))) #helytelen valaszok szazalekosan
        print("Number of COS equalities: " + str(equalityCounter)) #egyenlosegek




def filterData(list): #train data filterezes
    filteredList = []
    stopchars = ".,$?!€;-:/%(){}[]^`\'\\\""
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

def avg_feature_vector(words,num_features, index2word_set):
    #hatha megis talalunk valamit, amit meg nem szurtunk ki
    stop_words = set(stopwords.words('english')) #stopwordok
    stop_words.add("n\'t")
    stop_words.add("\'s")
    stop_words.add("\'ll")
    stop_words.add("\'d")
    stop_words.add("\'m")
    stop_words.add("\'ve")
    stop_words.add("\'")
    stop_words.add("will")
    stop_words.add("would")
    #felveszunk egy 0 vectort
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in stop_words:
            continue
        nwords = nwords+1
        featureVec = np.add(featureVec, index2word_set[word]) #kinyerjuk a modelbol a szohoz tartozo vectort es beletesszuk a featureVecbe

    if nwords>0:
        featureVec = np.divide(featureVec, nwords) #atlagoljuk a vektort a szavak szamaval
    return featureVec


def readTrainingData(filename): #fuggveny trainDataBeolvasasara
    df=pd.read_csv(filename) #open file
    s1=df['sentence1'].values.tolist() #read columns
    s1 = filterData(s1)
    s2=df['sentence2'].values.tolist()
    s2 = filterData(s2)
    s3=df['sentence3'].values.tolist()
    s3 = filterData(s3)
    s4=df['sentence4'].values.tolist()
    s4 = filterData(s4)

    c = s1 + s2 + s3 + s4 # a beolvasott oszlopokat konkatenaljuk a corpusba
    return c


def readTrainingDataB(filename): #fuggveny trainDataBeolvasasara
    df=pd.read_csv(filename) #open file
    s1=df['InputSentence1'].values.tolist() #read columns
    s1 = filterData(s1)
    s2=df['InputSentence2'].values.tolist()
    s2 = filterData(s2)
    s3=df['InputSentence3'].values.tolist()
    s3 = filterData(s3)
    s4=df['InputSentence4'].values.tolist()
    s4 = filterData(s4)
    r1=df['RandomFifthSentenceQuiz1'].values.tolist()
    r1 = filterData(r1)
    r2=df['RandomFifthSentenceQuiz2'].values.tolist()
    r2 = filterData(r2)
    c = s1 + s2 + s3 + s4 + r1 + r2 # a beolvasott oszlopokat konkatenaljuk a corpusba
    return c


c1 = readTrainingData("rcs17.csv")
c2 = readTrainingDataB("rcs_test.csv")
corpus = c1 + c2

data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
#print(data)

model = Doc2Vec(data, alpha=0.025, min_alpha=0.025, vector_size=100, window=2, min_count=1, workers=4)

t1 = Testing("rcs_test.csv",model)
t1.calculateAccuracy()
