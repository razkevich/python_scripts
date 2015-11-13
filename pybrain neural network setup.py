# -*- coding: utf-8 -*-
"""
Created on Mon Sep 07 10:05:57 2015

@author: razkevich
"""
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
import pybrain
from pybrain.tools.shortcuts import buildNetwork

from pybrain.supervised.trainers import BackpropTrainer

from pybrain.datasets import SupervisedDataSet
net = buildNetwork(X_train.shape[1], 3, 1, bias=True)
ds = SupervisedDataSet(X_train.shape[1], 1)

for i in range(1,len(X_train.toarray())):
    ds.addSample(totuple(X_train[i].toarray()[0]),Y_train.iloc[i]==3)
    print i
trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence()

