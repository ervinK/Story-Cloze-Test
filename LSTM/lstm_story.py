import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def vectorGen(down, up):

    arrs = [[np.array([[i, i+1, i+2], [i+3, i+4, i+5]]) for i in range(down, up)]] # 0-100
    return arrs

def addToVec(vec1):
    featureVec = np.zeros((100,2,3,), dtype="int")
    for x in vec1:
        featureVec = np.add(featureVec, x)
    return featureVec

vec1 = vectorGen(0,100)
feature1 = addToVec(vec1)

vec2 = vectorGen(3,103)
feature2 = addToVec(vec2)

vec_val1 = vectorGen(100,200)
feature_val1 = addToVec(vec_val1)

vec_val2 = vectorGen(103,203)
feature_val2 = addToVec(vec_val2)


model = Sequential()
model.add(LSTM(100, input_shape=(2, 3),return_sequences=True))
model.add(Dense(3))
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.fit(feature1, feature2, nb_epoch=10000, batch_size=1, verbose=2,validation_data=(feature_val1, feature_val2))

predict = model.predict(feature1)
print(predict)
