# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from Regression import RegressionBase, LogisticRegression

class Training(object):

    def __init__(self, trX, trY, numSteps, learning_rate, regression = RegressionBase):
        self.trX = trX
        self.trY = trY
        self.numSteps = numSteps
        self.learning_rate = learning_rate
        self.cost = regression.loss(self.trY)

    def gradient(self):
        self.g_w = T.grad(cost = self.cost, )
