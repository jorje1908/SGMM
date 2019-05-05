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
warnings.filterwarnings("ignore", category=DeprecationWarning)

from supervisedGmm import SupervisedGMM
from loaders2 import loader
from experFuncs import experiment1
#from experFuncs import transduction
#from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
#kmeansLogRegr
#import time
#from metricsFunctions import calc_metrics, metrics_cluster, optimalTau


np.random.seed( seed = 0)



#LOAD DATA
file1 = '/home/george/github/sparx/data/sparcs00.h5'
file2 = '/home/george/github/sparx/data/sparcs01.h5'
data, dataS, idx = loader(2000, 300, file1, file2)


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
##Setting SGMM Parameteres

alpha = [ 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000 ]
n_clusters = 4
cv = 10
scoring = 'neg_log_loss'
mcov = 'diag'
adaR = 1

#INITIALIZING THE MODEL
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = 10, tol = 10**(-6),
                         max_iter = 100, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 1, verbose = 0, scoring = scoring,
                         cv = cv)


#SET AVERAGING AND TRANSDUCTION PARAMETERS AND DATASETS
avg = 2
trans = 2
tr_sz = 0.25
X = data.iloc[:,0:-1].values
Y = data.iloc[:,-1].values

myDict = experiment1(X, Y, model,
                     averaging = avg, trans = trans, train_size = tr_sz)



#Directory to Save the data"
Directory = "Results/sparx/"
np.save( Directory+"sparxSGMMD.npy", myDict)

pdTest = pd.DataFrame(myDict['testF'], columns = columns)
pdTrain = pd.DataFrame(myDict['trainF'], columns = columns)

pdTest.to_csv(Directory+"testSpSGMM.csv", index = False, float_format = '%.3f')
pdTrain.to_csv(Directory+"trainSpSGMM.csv", index = False, float_format = '%.3f')













