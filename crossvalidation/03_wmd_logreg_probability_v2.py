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
from sklearn.tree import DecisionTreeClassifier

def format(proba_test):
    probabs = []
    for x in range(0,len(proba_test)):
        probabs += str(proba_test[x]).split(" ")
    retProbabs = []
    for x in probabs:
        tmp = x.replace('[','')
        toAdd = tmp.replace(']','')
        retProbabs.append(toAdd)
    return retProbabs

def takeDecisions(probabs):
    decisionArray = []
    i = 0
    while i < len(probabs):
        first1 = probabs[i]
        first2 = probabs[i+1]
        second1 = probabs[i+2]
        second2 = probabs[i+3]
        if first2 > second2:
            decisionArray.append(1)
            decisionArray.append(0)
        else:
            decisionArray.append(0)
            decisionArray.append(1)
        i+=4
    return decisionArray


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
#print(len(LogReg.decision_function(X)))

#y_pred = LogReg.predict(X_test)



#print(classification_report(y, y_pred))

#print(LogReg.classes_)
proba_test = LogReg.predict_proba(X_test)


probabs = format(proba_test)

decisionArray = takeDecisions(probabs)

#print(len(decisionArray))
#print(len(y_test))

count = 0
falseNegativeCount = 0
falsePositiveCount = 0
i = 0
while i < len(decisionArray):
    if decisionArray[i] == y_test[i] and decisionArray[i+1] == y_test[i+1]:
        count += 1
    elif decisionArray[i] == 0 and y_test[i] == 1:
        falseNegativeCount += 1
    elif decisionArray[i] == 1 and y_test[i] == 0:
        falsePositiveCount += 1
    i += 2

print("Accuracy: " + str(count/(len(decisionArray)/2)))
print("False Positive: " + str(falsePositiveCount/(len(decisionArray)/2)))
print("False Negative: " + str(falseNegativeCount/(len(decisionArray)/2)))
