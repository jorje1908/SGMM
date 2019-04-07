#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:10:04 2019

@author: george
"""

import numpy as np
import pandas as pd



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from supervisedGmm import SupervisedGMM
from metricsFunctions import calc_metrics, metrics_cluster, optimalTau
#from superGmmMother import superGmmMother
from loaders2 import loader
from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
kmeansLogRegr

np.random.seed( seed = 0)
###############################################################################

#READING DATA SETTING COLUMNS NAMES FOR METRICS
file1 = '/home/george/github/sparx/data/sparcs00.h5'
file2 = '/home/george/github/sparx/data/sparcs01.h5'
data, dataS, idx = loader(40000, 300, file1, file2)


cols = data.columns
colA = cols[761:1100]
data = data.drop(colA, axis = 1)

columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
###############################################################################

##Fitting SGMM
Cs = [  10 ]
model = SupervisedGMM( Cs = Cs, n_clusters = 12, max_iter2 = 20, tol = 10**(-6))
Xtrain, Xtest, ytrain, ytest = model.split( data = data.values)
model = model.fit( Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain)

mTest, mTrain = model.mTest, model.mTrain
logisRegre = model.LogRegr
fitP = model.fitParams
labTrain, labTest = fitP['labTrain'], fitP['labTest']


probTest, probTrain = model.predict_prob_int( Xtest = Xtest, Xtrain = Xtrain )
tau = optimalTau(probTrain, ytrain)
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

###############################################################################