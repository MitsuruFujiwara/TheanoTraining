# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from keras.models import Sequential
model = Sequential()

from keras.layers import Dense, Activation
model.add(Dense(output_dim=2, input_dim=4))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
model.add(Activation("softmax"))

model.compile(loss='mean_squared_error', optimizer='sgd')

df = pd.read_csv("test_data.csv")
trX = df[["X1", "X2", "X3", "X4"]]
trY = df["Y"]

model.fit(trX, trY, nb_epoch=100)
