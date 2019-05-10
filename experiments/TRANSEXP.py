#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:42:57 2019

@author: george
"""

import sys

sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from supervisedGmm import SupervisedGMM
from metricsFunctions import optimalTau, calc_metrics, metrics_cluster
import time
import pandas as pd
from experFuncs import transduction, experiment1




np.random.seed( seed = 0 )
######### TRANSDUCTION EXPERIMENT ############################################
# 2 GAUSSIANS TO GENERATE DATA AND THEN GENERATE LABELS FROM ANOTHER
# TWO GAUSSIANS WITH THEIR CORRESPONDING  WEIGHTS


#GENERATE A LIST OF WHICH GAUSSIAN WILL GENERATE THE POINTS

#POINTS
N = 10000
mix = [0.45, 0.55]

index = np.random.choice( [1, 2], size = N, p = mix )
index1 = np.where( index == 1)[0]
index2 = np.where( index == 2)[0]

#GAUSSIAN 1
m1 = [-5, 5]
S1 = np.array( [[0.5,0], [0,0.7]])
w1 = np.array([5,5])

#GAUSSIAN 2
m2 = [5, 5]
S2 = np.array( [[0.8, 0], [0, 0.4]])
w2 = np.array([-5, 5] )


#GENERATE  DATA
ones = np.ones([N,1])
X = np.zeros( [N, 2]) 
Y = np.zeros( N )

for i in np.arange( N ):
    
    if index[i] == 1:
        
        X[i] = np.random.multivariate_normal( m1, S1 )
        Y[i] = np.sum (X[i] * w1 )
        
    else:
    
         X[i] = np.random.multivariate_normal( m2, S2 )
         Y[i] = np.sum (X[i] * w2 )
    Y[i] = np.sign( Y[i] )
    if Y[i] < 0:
        Y[i] = 0

      
ind0 = np.where( Y == 0)[0]
ind1 = np.where( Y == 1)[0]
     
#PLOTTTING GAUSSIANS PLUS    POSITIVE NEGATIVE EXAMPLES
fig, ax = plt.subplots(1,2, figsize = [13,5])         
ax[0].scatter( X[index1 ,0], X[index1, 1]) 
ax[0].scatter( X[index2 ,0], X[index2, 1])   
ax[1].scatter( X[ind0 ,0], X[ind0, 1]) 
ax[1].scatter( X[ind1 ,0], X[ind1, 1])  
 
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')
ax[0].legend(['G1', 'G2'])  
ax[1].legend([ 'Positive', 'Negative'])


columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
###############################################################################
X = np.concatenate((ones, X), axis = 1) 
##Fitting SGMM
adaR = 1
alpha = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 ]
#alpha = [0.00001]
#alpha = [1]
n_clusters = 4
vrb = 0
cv = 10
warm = 0
scoring = 'neg_log_loss'
mcov = 'diag'
mx_it = 1000
mx_it2 = 10
km = 1
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-3),
                         max_iter = mx_it, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 1, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-2) )

#SPLIT THE DATA
trans = 14
#Xtrain, Xtest, ytrain, ytest = model.split( X = X, y = Y, split = 0.25)
gausDict = experiment1(X, Y.astype( int ), model, trans = trans)

#FIT THE MODEL 
#start = time.time()
#model = model.fit( Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain, kmeans = km)
#end = time.time() - start
#print( " Algorith run in {}s".format( end ))
#
##GET THE MEMBERSHIPS, LOGISTIC REGRESSION FIT PARAMETERS
#mTest, mTrain = model.mTest, model.mTrain
#logisRegre = model.LogRegr
#fitP = model.fitParams
#labTrain, labTest = fitP['labTrain'], fitP['labTest']
#
#
##PREDICT THE INTERNAL PROBABILITIES 
#probTest, probTrain = model.predict_prob_int( Xtest = Xtest, Xtrain = Xtrain )
##CALCULATE THE OPTIMAL TAU
#tau = optimalTau(probTrain, ytrain)
##tau = 0.5
#metTest,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, y = ytest)
#metTrain ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau, y = ytrain)
#
##TOTAL METRICS OF SGMM (PANDA MATRICES)
#metTestSGMM = pd.DataFrame( [metTest], columns = columns)
#metTrainSGMM = pd.DataFrame( [metTrain], columns = columns)
#
#means = model.means
#cov = model.cov
#weights = model.weights
#
##CLUSTER METRICS OF SGMM (PANDA MATRICES)
#metCTrainSGMM, metCTestSGMM = metrics_cluster(models = logisRegre, 
#                                              ytrain = ytrain, ytest = ytest,
#                                              testlabels = labTest,
#                                              trainlabels = labTrain,
#                                              Xtrain = Xtrain, Xtest = Xtest)



###############################################################################






