# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv('test_data.csv')

# set header
c = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9','X10', 'X11', 'X12',\
'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21','X22', 'X23',\
'X24', 'X25', 'X26', 'X27']
trX = np.array(df[c].fillna(0))
trY = np.array(df['Y'].fillna(0))

# set moedel
model = Sequential()
model.add(Dense(output_dim=10, input_dim=trX.shape[1]))
model.add(Activation('tanh'))
model.add(Dense(output_dim=3, input_dim=10))
model.add(Activation('relu'))
model.add(Dense(output_dim=1, input_dim=3))
model.add(Activation('softmax'))

# training
model.compile(loss='mean_squared_error', optimizer='sgd')

his  = model.fit(trX, trY, nb_epoch=10000)
loss = his.history['loss']
plt.plot(loss)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
