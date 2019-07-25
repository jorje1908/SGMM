#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:10:04 2019

@author: george
"""

import sys
sys.path.append('../SGMM')
sys.path.append('../metrics')


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
from supervisedGmm_Clean import SupervisedGMM as SGMM
from metricsFunctions import sgmmResults
#from superGmmMother import superGmmMother
from loaders2 import loader
#from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
#kmeansLogRegr

np.random.seed( seed = 0)
###############################################################################

#READING DATA SETTING COLUMNS NAMES FOR METRICS
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

##Fitting SGMM
#alpha = [1]
n_clusters = 4
cv = 2
scoring = 'neg_log_loss'
mcov = 'diag'
mx_it2 = 10
mx_it = 1000
warm = 0
vrb = 0
adaR = 0
km = 1
model1 = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-5),
                         max_iter = mx_it, mcov = mcov, adaR = adaR,
                         transduction = 0, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-2) )

model = SGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-5),
                         max_iter = mx_it, mcov = mcov, adaR = adaR,
                         transduction = 0, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-3) )

#SPLIT THE DATA
Xtrain, Xtest, ytrain, ytest = model1.split( data = data.values)

#FIT THE MODEL 
start = time.time()
model = model.fit( Xtrain = Xtrain, Xtest = [], ytrain = ytrain,
                  mod = 1, kmeans = km, simple = 1)
end = time.time() - start
print( " Algorith run in {}s".format( end ))

probTest = model.predict_proba( Xtest )
probTrain = model.predict_proba( Xtrain )



res = sgmmResults(model, probTest, probTrain, ytest, ytrain)
monitor = model.monitor_