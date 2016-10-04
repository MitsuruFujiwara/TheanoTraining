# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

class RegressionBase(object):
    """
    Class for Linear Regression
    """

    def __init__(self, trX, trY, numSteps, learning_rate):
        # training data
        self.trX = np.asarray(trX, dtype = theano.config.floatX)
        self.trY = np.asarray(trY, dtype = theano.config.floatX)
        
        # training parameters
        self.numSteps = numSteps
        self.learning_rate = learning_rate
        
        self.numData = self.trX.shape[0] # number of dataset
        self.numFactor = self.trX.shape[1] # number of factor
        
        # shared parameters
        self.W = theano.shared(np.zeros(self.numFactor), name = "W")
        self.b = theano.shared(0., name = "b")
        
        self.X = T.vector()
        self.Y = T.scalar()
        
        # set gradient
        self.g_w = T.grad(cost = self.loss(), wrt = self.W)
        self.g_b = T.grad(cost = self.loss(), wrt = self.b)
        
        # set train function
        self.updates = [(self.W, self.W - self.learning_rate * self.g_w),(self.b, self.b - self.learning_rate * self.g_b)]
        self.train = theano.function(inputs = [self.X, self.Y], outputs = self.loss(), updates = self.updates)

    def inference(self):
        return T.dot(self.X, self.W) + self.b
    
    def loss(self):
        return T.mean(T.sqr(self.inference() - self.Y))

    def run(self):
        for i in range(self.numSteps):
            for x, y in zip(self.trX, self.trY):
                self.train(x, y)
            print self.W.get_value(), self.b.get_value()
            
if __name__ == '__main__':
    df = pd.read_csv("test_data.csv")
    trX = df[["X1", "X2", "X3"]]
    trY = df["Y"]
    
    numSteps = 100
    learning_rate = 0.001
    
    rb = RegressionBase(trX, trY, numSteps, learning_rate)
    rb.run()
    