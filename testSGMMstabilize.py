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
#import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
from supervisedGmm import SupervisedGMM
from metricsFunctions import calc_metrics, metrics_cluster, optimalTau
#from superGmmMother import superGmmMother
from loaders2 import loader
#from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
#kmeansLogRegr

np.random.seed( seed = 0)
###############################################################################

#READING DATA SETTING COLUMNS NAMES FOR METRICS
file1 = '/home/george/github/sparx/data/sparcs00.h5'
file2 = '/home/george/github/sparx/data/sparcs01.h5'
data, dataS, idx = loader(15000, 300, file1, file2)


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
Cs = [  10 ]
alpha = [ 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000 ]
#alpha = [1]
n_clusters = 4
cv = 5
scoring = 'neg_log_loss'
mcov = 'full'
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 =10, tol = 10**(-6),
                         max_iter = 30, alpha = alpha, mcov = mcov, adaR = 0,
                         transduction = 1, verbose = 1, scoring = scoring,
                         cv = cv)

#SPLIT THE DATA
Xtrain, Xtest, ytrain, ytest = model.split( data = data.values)

#FIT THE MODEL 
start = time.time()
model = model.fit( Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain)
end = time.time() - start
print( " Algorith run in {}s".format( end ))

#GET THE MEMBERSHIPS, LOGISTIC REGRESSION FIT PARAMETERS
mTest, mTrain = model.mTest, model.mTrain
logisRegre = model.LogRegr
fitP = model.fitParams
labTrain, labTest = fitP['labTrain'], fitP['labTest']


#PREDICT THE INTERNAL PROBABILITIES 
probTest, probTrain = model.predict_prob_int( Xtest = Xtest, Xtrain = Xtrain )
#CALCULATE THE OPTIMAL TAU
tau = optimalTau(probTrain, ytrain)
#tau = 0.5
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