# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

class RegressionBase(object):
    """
    Class for Simmple Linear Regression
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

    def inference(self):
        # set predict function
        return T.dot(self.X, self.W) + self.b

    def loss(self):
        # set loss function
        return T.mean(T.sqr(self.inference() - self.Y))

    def train_model(self):
        # set model for training
        g = self.gradient()
        updt = [(self.W, self.W - self.learning_rate * g[0]),(self.b, self.b - self.learning_rate * g[1])]
        self.train = theano.function(inputs = [self.X, self.Y], outputs = self.loss(), updates = updt)

    def gradient(self):
        self.g_w = T.grad(cost = self.loss(), wrt = self.W)
        self.g_b = T.grad(cost = self.loss(), wrt = self.b)
        return (self.g_w, self.g_b)

    def run(self):
        self.train_model()
        for i in range(self.numSteps):
            for x, y in zip(self.trX, self.trY):
                self.train(x, y)
            print self.W.get_value(), self.b.get_value()

        return (self.W.get_value(), self.b.get_value())

class LogisticRegression(RegressionBase):
    """
    Class for Logistic Regression
    """

    def __init__(self, trX, trY, numSteps, learning_rate, numClass):
        RegressionBase.__init__(self, trX, trY, numSteps, learning_rate)

        self.trY = np.asarray(trY, dtype = 'int32') # change dtype
        self.numClass = numClass # number of output class

        self.W = theano.shared(np.zeros((self.numFactor, self.numClass)), name = "W")
        self.b = theano.shared(np.zeros((self.numClass,)), name = "b")

        self.X = T.dmatrix("X")
        self.Y = T.lvector("Y")

    def inference(self):
        return T.nnet.softmax(T.dot(self.X, self.W) + self.b)

    def loss(self):
        return -T.mean(T.log(self.inference())[T.arange(self.Y.shape[0]), self.Y])

    def run(self):
        self.train_model()
        for i in range(self.numSteps):
            for x, y in zip(self.trX, self.trY):
                x = x.reshape((self.numClass, self.numFactor))
                y = y.reshape((self.numClass,))
                print y
                self.train(x, y)
            print i, self.W.get_value(), self.b.get_value()

        return (self.W.get_value(), self.b.get_value())

if __name__ == '__main__':
    df = pd.read_csv("test_data.csv")
    trX = df[["X1", "X2", "X3", "X4"]]
    trY = df["Y"]

    numSteps = 100
    learning_rate = 0.001

    rb = RegressionBase(trX, trY, numSteps, learning_rate)
    lr = LogisticRegression(trX, trY, numSteps, learning_rate, 1)
    lr.run()
