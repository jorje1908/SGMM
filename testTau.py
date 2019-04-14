#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:29:52 2019

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
#from superGmmMother import superGmmMother
from loaders2 import loader
from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
kmeansLogRegr

from visualFunctions import CreateClouds, CreateCloudsWeights, plot_parallel,\
findbinary, heatmap


np.random.seed( seed = 0)
###############################################################################

#READING DATA SETTING COLUMNS NAMES FOR METRICS
file1 = '/home/george/github/sparx/data/sparcs00.h5'
file2 = '/home/george/github/sparx/data/sparcs01.h5'
data, dataS, idx = loader(5000, 300, file1, file2)


cols = data.columns
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
Cs = [  10 ]
alpha = [0.1, 0.0001, 2, 0.001]
model = SupervisedGMM( Cs = Cs, n_clusters = 2, max_iter2 = 5, tol = 10**(-6),
                                                              max_iter = 5,
                                                              alpha = alpha,
                                                              mcov = 'diag')

Xtrain, Xtest, ytrain, ytest = model.split( data = data.values)
start = time.time()
model = model.fit( Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain)
end = time.time() - start
print( " Algorith run in {}s".format( end ))


mTest, mTrain = model.mTest, model.mTrain
logisRegre = model.LogRegr
fitP = model.fitParams
labTrain, labTest = fitP['labTrain'], fitP['labTest']


probTest, probTrain = model.predict_prob_int( Xtest = Xtest, Xtrain = Xtrain )
metricstau = optimalTau(probTrain, ytrain, returnAll = 1, targetValue = 1)
tau  =  metricstau['tau']
tau = metricstau['tauTarget']
metTest,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, y = ytest)
metTrain ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau, y = ytrain)

#CLUSTER METRICS OF SGMM (PANDA MATRICES)
metCTrainSGMM, metCTestSGMM = metrics_cluster(models = logisRegre, 
                                              ytrain = ytrain, ytest = ytest,
                                              testlabels = labTest,
                                              trainlabels = labTrain,
                                              Xtrain = Xtrain, Xtest = Xtest)
#TOTAL METRICS OF SGMM (PANDA MATRICES)
metTestSGMM = pd.DataFrame( [metTest], columns = columns)
metTrainSGMM = pd.DataFrame( [metTrain], columns = columns)
