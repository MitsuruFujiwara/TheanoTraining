# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import Randgen

class RegressionBase(object):
    """
    Simple Linear Regression Class

    Estimate parameters w and b from training data trX and trY
    """

    def __init__(self, trX, trY, numStep, learning_rate):
        # training data
        self.trX = theano.shared(np.asarrat(trX))
        self.trY = theano.shared(np.asarrat(trY))

        self.numStep = numStep # number of training
        self.learning_rate = learning_rate # learning rate for training

        # parameters
        self.n = len(trY)
        self.m = 1 #TBD
        self.w = theano.shared(np.zeros((self.m, 1), dtype = theano.config.floatX))
        self.b = theano.shared(0.0)
        self.params = (self.w, self.b)

    def inference(self):
        return T.dot(self.trX, self.w) + self.b

    def loss(self):
        # loss function
        return T.mean(T.pow(self.trY - self.inference(), 2.0))

    def train(self):
        # Set Gradient
        self.grad_w =





if __name__ == '__main__':
    n = 100 # number of random variable in single data set
    m = 1 # number of data sets
    sigma = 0.03 # volatility of data set

    r = Randgen.Randgen(m, n, sigma)

    trX = r.sin_wave_y()
    rb = RegressionBase()
