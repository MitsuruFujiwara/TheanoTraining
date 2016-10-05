# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

class RegressionBase(object):

    """
    Base Class for Regression
    """

    def __init__(self, trX, numInput, numOutput):

        """
        trX: input data
        numInput: number of input units
        numOutput: number of output units
        """

        self.trX = trX
        self.w = w
        self.b = b
        self.params = (self.w, self.b)

    def inference(self):
        # predict function
        return T.dot(self.trX, self.w) + self.b

    def loss(self, y):
        # return mean squared error for target value y
        return T.mean(T.pow(y - self.inference(), 2.0))


class LogisticRegression(RegressionBase):

    """
    Class for Logistic Regression
    """

    def __init__(self, trX, numInput, numOutput):
        RegressionBase.__init__(self, trX, numInput, numOutput)

    def inference(self):
        # predict function
        return T.nnet.softmax(T.dot(self.trX, self.w) + self.b)

    def loss(self, y):
        # return negative log-likehood function
        return -T.mean(T.log(self.inference())[T.arange(y.shape[0]), y])


class RectifiedLinearRegression(RegressionBase):
    
    """
    Class for Rectified Linear Regression
    """
    
    def __init__(self, trX, numInput, numOutput):
        RegressionBase.__init__(self, trX, numInput, numOutput)
            
    def inference(self):
        # predict function
        return T.nnet.relu(T.dot(self.trX, self.w) + self.b)
        
