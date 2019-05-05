#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:03:47 2019

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
from metricsFunctions import calc_metrics, metrics_cluster, optimalTau
from loaders2 import loader
from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
kmeansLogRegr, xboost, gradboost
from sklearn.model_selection import train_test_split
import time

np.random.seed( seed = 0 )
###############################################################################

#READING DATA SETTING COLUMNS NAMES FOR METRICS
file1 = '/home/george/github/sparx/data/sparcs00.h5'
file2 = '/home/george/github/sparx/data/sparcs01.h5'
data, dataS, idx1 = loader(2000, 300, file1, file2)

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

methods = ['L1', 'NN', 'RF', 'Ada', 'GB']
###############################################################################

averaging = 2
Cs = [ 0.01, 0.01, 1, 10, 100, 1000 ]
alpha = [0.1, 0.01, 0.001, 0.0001, 10**(-7), 1 ]
train_size = 0.25

X = data.iloc[:,0:-1].values
Y = data.iloc[:,-1].values
idx = np.arange(X.shape[0])
index100 = []

start = time.time()
trainRes = np.zeros([ 5,16 ])
testRes = np.zeros([ 5,16 ])
###############################################################################
for i in np.arange( averaging ):
     print("\n################################################")
     print("ITERATION: {} OF AVERAGING".format( i))
        
     #SPLIT DATA INTO TRAINING AND TEST
     Xtrain, Xtest, ytrain, ytest, itr, itst = train_test_split( X, Y, idx,
                                                       train_size = train_size,
                                                       stratify = Y,
                                                       random_state = i)
        
        
     index100.append( itr[100] )
     #FITTING L1 LOGISTIC REGRESSION
     pL1, probTestL1, probTrainL1 = logisticRegressionCv2( Xtrain = Xtrain,
                                                          ytrain = ytrain,
                                                          Xtest = Xtest,
                                                    ytest = ytest, Cs = Cs )
###############################################################################
     print("\n################################################")
     print("ITERATION: {} RUNNING L1 LOGISTIC REGRESSION".format( i))
     #METRICS L1 LOGISTIC REGRESSION  
     tau = optimalTau(probTrainL1, ytrain)

     metTestL1,_ = calc_metrics(custom_prob = probTestL1.copy(), tau = tau, y = ytest)
     metTrainL1 ,_ = calc_metrics(custom_prob = probTrainL1.copy(), tau = tau, y = ytrain)

     trainRes[0,:] += metTrainL1
     testRes[0,:] += metTestL1
    
    
###############################################################################
     print("\n################################################")
     print("ITERATION: {} RUNNING NEURAL NETWORKS".format( i))
     #Fitting Neural Nets
     pNN, probTestNN, probTrainNN = neural_nets( Xtrain = Xtrain,
                                                  ytrain = ytrain,
                                                  Xtest = Xtest,
                                                  ytest = ytest,
                                                  h_l_s = (4 ,4, 2))

     #Metrics Neurals Nets
     tau = optimalTau(probTrainNN, ytrain)

     metTestNN,_ = calc_metrics(custom_prob = probTestNN.copy(), tau = tau, y = ytest)
     metTrainNN ,_= calc_metrics(custom_prob = probTrainNN.copy(), tau = tau, y = ytrain)
     
     trainRes[1,:] += metTrainNN
     testRes[1,:] += metTestNN
    
###############################################################################



###############################################################################
     print("\n################################################")
     print("ITERATION: {} RUNNING RANDOM FORESTS".format( i))
     #RANDOM FORESTS
     params, probTest, probTrain = randomforests(Xtrain = Xtrain, ytrain = ytrain,
                                            Xtest = Xtest, ytest = ytest)

     tau = optimalTau(probTrain, ytrain)
     metTestRF,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, y = ytest)
     metTrainRF ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau, y = ytrain)
     
     trainRes[2,:] += metTrainRF
     testRes[2,:] += metTestRF

    


###############################################################################
     print("\n################################################")
     print("ITERATION: {} RUNNING ADA BOOST".format( i))
    #Ada boost
     params, probTest, probTrain = xboost(Xtrain = Xtrain, ytrain = ytrain,
                                            Xtest = Xtest, ytest = ytest)

     tau = optimalTau(probTrain, ytrain)
     metTestAda,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, y = ytest)
     metTrainAda ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau, y = ytrain)
     
     trainRes[3,:] += metTrainAda
     testRes[3,:] += metTestAda



###############################################################################
     print("\n################################################")
     print("ITERATION: {} RUNNING GRAD BOOST".format( i))
    #Grad boost
     params, probTest, probTrain = gradboost(Xtrain = Xtrain, ytrain = ytrain,
                                            Xtest = Xtest, ytest = ytest)

     tau = optimalTau(probTrain, ytrain)
     metTestGB,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, y = ytest)
     metTrainGB ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau, y = ytrain)

     trainRes[4,:] += metTrainGB
     testRes[4,:] += metTestGB

end = time.time() - start
###############################################################################
print("################################################")
print("\n END OF AVERAGING- TIME ELAPSED: {}".format(end) )

trainResPD = pd.DataFrame( trainRes, index = methods, columns = columns)/averaging
testResPD = pd.DataFrame( testRes, index = methods, columns = columns)/averaging









