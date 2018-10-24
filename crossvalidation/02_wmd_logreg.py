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


trainAddress = 'train_wmd.csv'
testAddress = 'test_wmd.csv'

train = pd.read_csv(trainAddress)
test = pd.read_csv(testAddress)
train.columns = ['InputSentence1', 'InputSentence2',	'InputSentence3', 'InputSentence4', 'AnswerRightEnding', 'sumOfWmd']
test.columns = ['InputSentence1', 'InputSentence2',	'InputSentence3', 'InputSentence4', 'AnswerRightEnding', 'sumOfWmd']

y = train.ix[:,4].values #labelnek megadjuk a 9. oszlopot
y_test = test.ix[:,4].values
#print(y)
sentences = train.ix[:,(0,1,2,3,5)].values #felvesszuk az 5. es a 11. oszlopot
test_sentences = test.ix[:,(0,1,2,3,5)].values

#print(sentences)
#valid_data = valid.ix[:,(5,11)].values

X = scale(sentences) #az 5. es a 11. oszlopot elhelyezzük a koordinatarendszerben
X_test = scale(test_sentences)


LogReg = LogisticRegression() #felvesszuk a sigmoidot
LogReg.fit(X,y) #fiteljuk a sigmoidra

#print(LogReg.score(X,y))

y_pred = LogReg.predict(X)



#print(classification_report(y, y_pred))

y_testPred = LogReg.predict(X_test)

#print(y_test)
#print(y_testPred)

countTruePositiveAll = 0
for x in range(0,len(y_test)):
    if y_test[x] == y_testPred[x]:
        countTruePositiveAll += 1

countTruePositiveRight = 0
for x in range(0,len(y_test)):
    if y_test[x] == 1 and  y_testPred[x] == 1:
        countTruePositiveRight += 1

countTruePositiveFalse = 0
for x in range(0,len(y_test)):
    if y_test[x] == 0 and  y_testPred[x] == 0:
        countTruePositiveFalse += 1

countFalsePositiveTrue = 0
for x in range(0,len(y_test)):
    if y_test[x] == 0 and  y_testPred[x] == 1:
        countFalsePositiveTrue += 1

countFalsePositiveFalse = 0
for x in range(0,len(y_test)):
    if y_test[x] == 1 and  y_testPred[x] == 0:
        countFalsePositiveFalse += 1

print("True pozitiv 0,1 osztalyozas: " + str(countTruePositiveAll/len(y_testPred)))
print("True pozitiv 1 osztalyozas: " + str(countTruePositiveRight/len(y_testPred)))
print("True negativ 0 osztalyozas: " + str(countTruePositiveFalse/len(y_testPred)))
print("Fals pozitiv 1 osztalyozas: " + str(countFalsePositiveTrue/len(y_testPred)))
print("Fals negativ 0 osztalyozas: " + str(countFalsePositiveFalse/len(y_testPred)))

#print(y_valid_pred)
#print(classification_report(y_valid_pred, y_pred))"""
