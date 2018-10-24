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



address = 'mtcars.csv'
address2 = 'mtcars_valid.csv'
cars = pd.read_csv(address)
valid = pd.read_csv(address2)
cars.columns = ['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
valid.columns = ['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']

y = cars.ix[:,9].values #labelnek megadjuk a 9. oszlopot
cars_data = cars.ix[:,(5,11)].values #felvesszuk az 5. es a 11. oszlopot
valid_data = valid.ix[:,(5,11)].values

X = scale(cars_data) #az 5. es a 11. oszlopot elhelyezzük a koordinatarendszerben
x_valid = scale(valid_data)

LogReg = LogisticRegression() #felvesszuk a sigmoidot
LogReg.fit(X,y) #fiteljuk a sigmoidra

print(LogReg.score(X,y))

y_pred = LogReg.predict(X)

from sklearn.metrics import classification_report

print(classification_report(y, y_pred))

y_valid_pred = LogReg.predict(x_valid)

print(y_valid_pred)
#print(classification_report(y_valid_pred, y_pred))
