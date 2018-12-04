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

"""
Ugy dolgozzuk fel az eredeti train datat, hogy ugyanannyi befejezes legyen benne, mint sima mondat.
Hogy az allitasok minden fajtajat elsajatitsuk tanulaskor, ezert ha tobb mondatot olvastunk be 1-2-3-4 mondatonkent,
mint amennyi a befejezes, akkor utana mar csak befejezest irunk ki. Igy megmarad az egyensuly tanulaskor
"""


with open("end_or_not_train_data.csv", 'w', newline='') as f:
    train = pd.read_csv("original_train_data.csv")

    s1 = train['sentence1']
    s2 = train['sentence2']
    s3 = train['sentence3']
    s4 = train['sentence4']
    ending = train['ending']

    exportFieldNames = ['sentence', 'isEnding']
    noEnding = 0
    writer = csv.DictWriter(f, fieldnames=exportFieldNames)
    writer.writeheader()
    simpleSentenceCount = 0
    for iter in range(0, len(s1)):
        if(simpleSentenceCount < len(ending)):
            writer.writerow({'sentence' : s1[iter], 'isEnding' : 0 })
            writer.writerow({'sentence' : s2[iter], 'isEnding' : 0 })
            writer.writerow({'sentence' : s3[iter], 'isEnding' : 0 })
            writer.writerow({'sentence' : s4[iter], 'isEnding' : 0 })
            simpleSentenceCount += 4
        writer.writerow({'sentence' : ending[iter], 'isEnding' : 1 })
