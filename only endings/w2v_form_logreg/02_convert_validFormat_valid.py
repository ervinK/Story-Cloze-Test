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
import csv


with open("val_isEnding.csv", 'w', newline='') as f:
    train = pd.read_csv("val_set.csv")

    s1 = train['RandomFifthSentenceQuiz1']
    s2 = train['RandomFifthSentenceQuiz2']
    ans = train['AnswerRightEnding']

    exportFieldNames = ['sentence', 'isEnding']

    writer = csv.DictWriter(f, fieldnames=exportFieldNames)
    writer.writeheader()
    for iter in range(0,len(ans)-int(len(ans)/2)):
        if ans[iter] == 1:
            writer.writerow({'sentence' : s1[iter], 'isEnding' : 1 })
            writer.writerow({'sentence' : s2[iter], 'isEnding' : 0 })
        else:
            writer.writerow({'sentence' : s2[iter], 'isEnding' : 1 })
            writer.writerow({'sentence' : s1[iter], 'isEnding' : 0 })
