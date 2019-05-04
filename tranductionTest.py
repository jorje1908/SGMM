#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:57:47 2019

@author: george
"""

import numpy as np
import pandas as pd



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
#import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
from supervisedGmm import SupervisedGMM
from metricsFunctions import calc_metrics, metrics_cluster, optimalTau
from experFuncs import experiment1, transduction
from loaders2 import loader
#from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
#kmeansLogRegr

np.random.seed( seed = 0)




file1 = '/home/george/github/sparx/data/sparcs00.h5'
file2 = '/home/george/github/sparx/data/sparcs01.h5'
data, dataS, idx = loader(10000, 300, file1, file2)


cols = data.columns
#drop drgs and length of stay
colA = cols[761:1100]
colB = cols[0]
data = data.drop(colA, axis = 1)
data = data.drop(colB, axis = 1)
colss = data.columns.tolist()

columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
###############################################################################
##################### TESTING TRANSDACTION ####################################
##Fitting SGMM
Cs = [  10 ]
alpha = [ 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000 ]
#alpha = [1]
n_clusters = 4
cv = 10
scoring = 'neg_log_loss'
mcov = 'diag'
adaR = 1
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = 7, tol = 10**(-6),
                         max_iter = 100, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 1, verbose = 0, scoring = scoring,
                         cv = cv)

Xtrain, Xtest, ytrain, ytest = model.split( data = data.values, split = 0.75)


start = time.time()
print("Starting TransDuction")
resultsTest, resultsTrain = transduction(model, Xtrain, Xtest, ytrain, ytest, trans = 10, 
                       )


end =  time.time() - start
print("End of Transduction, time elapsed: {}".format( end))











