# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import Randgen

n = 1000
m = 100
k = 10
sigma = 0.03

r = Randgen.Randgen(m, n, sigma)
trY = list(r.sin_wave_y())[0]
trX = r.x[0]

X = T.scalar()
Y = T.scalar()

def model(X, w):
    return X * w

w = theano.shared(np.asarray(0., dtype = theano.config.floatX))
y = model(X, w)

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost = cost, wrt = w)
updates = [[w, w - gradient * 0.01]]

train = theano.function(inputs = [X, Y], outputs = cost, updates = updates, allow_input_downcast = True)

for i in range(10000):
    if i % 100 == 0:
        print str(i) + '  ' + str(w.get_value())
    for x, y in zip(trX, trY):
        train(x, y)
