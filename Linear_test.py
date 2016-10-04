# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

n = 100 # number of dataset
m = 1 # number of factor
sigma = 0.03 # volatility of y

trX = np.random.rand(n)
trY = 2 * trX + sigma * np.random.randn(n)

X = T.scalar()
Y = T.scalar()

W = theano.shared(0., name = "W")
b = theano.shared(0., name = "b")

inference = W * X + b
    
loss = T.mean(T.sqr(inference - Y))

g_w = T.grad(cost = loss, wrt = W)
g_b = T.grad(cost = loss, wrt = b)

learning_rate = 0.01
numSteps = 1000

train = theano.function(inputs = [X, Y], outputs = loss, \
updates = [(W, W - learning_rate * g_w),(b, b - learning_rate * g_b)])

for i in range(numSteps):
    for x, y in zip(trX, trY):
        train(x, y)
    if i % 100 == 0:
        print W.get_value(), b.get_value()