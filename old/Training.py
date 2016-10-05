# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from Regression import RegressionBase, LogisticRegression
from Randgen import Randgen

class Training(object):

    def __init__(self, trX, trY, numSteps, learning_rate, regression = RegressionBase):
        self.trX = theano.shared(np.asarray(trX, dtype = theano.config.floatX), borrow = True)
        self.trY = theano.shared(np.asarray(trY, dtype = theano.config.floatX), borrow = True)
        
        self.w = theano.shared(np.ones((trX.shape[1], trX.shape[1]), dtype=theano.config.floatX), name = "w", borrow=True)
        self.b = theano.shared(np.ones((trX.shape[1],), dtype=theano.config.floatX), name = "b", borrow=True)
        
        self.numSteps = numSteps
        self.learning_rate = learning_rate
        
        self.rg = regression(self.trX, trX.shape[1], trX.shape[1], self.w, self.b)
        self.cost = self.rg.loss(self.trY)
        
        # set gradient
        self.g_w = T.grad(cost = self.cost, wrt = self.rg.w)
        self.g_b = T.grad(cost = self.cost, wrt = self.rg.b)
        
        self.updates = [(self.rg.w, self.rg.w - self.learning_rate * self.g_w),\
        (self.rg.b, self.rg.b - self.learning_rate * self.g_b)]
        
    def train_model(self):
        return theano.function(inputs = [], outputs = self.cost, updates = self.updates)
    
    def train(self):
        for i in range(self.numSteps):
            current_cost = self.train_model()
            if i % 100 == 0:
                print i, self.w.get_value()[0][0], self.b.get_value()[0]
        
if __name__ == '__main__':
    n = 1
    m = 100
    r = Randgen(n, m, 0.03)
    y = np.array(list(r.linear_y()))
    x = np.array(r.x)
    
    numSteps = 1000
    learning_rate = 0.03
    
    t = Training(x, y, numSteps, learning_rate)
    t.train()
    