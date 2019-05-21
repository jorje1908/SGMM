#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 23:33:44 2019

@author: george
"""

import sys

#APPEND THE PATH FOR THE CODES USED
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import multivariate_normal
from supervisedGmm import SupervisedGMM
from metricsFunctions import optimalTau, calc_metrics, metrics_cluster,\
 sgmmResults
import time
import pandas as pd
from experFuncs import transduction, experiment1
from dataGen import genData

np.random.seed( seed = 0 )

covG = np.array( [[2,0], [0,2]])
covG2 = np.array( [[3,0], [0,3]])
enhance = 100

mix = [0.5, 0.5]
#GAUSS 1
m1 = [ -3, 3 ]
cov1 = covG
g1 = multivariate_normal(mean = m1, cov = cov1)

#GAUSS 2
m2 = [ 3, -3 ]
cov2 = covG
g2 = multivariate_normal(mean = m2, cov = cov2)

#GAUSS 3
m3 = [3, 3]
cov3 = covG2
g3 = multivariate_normal(mean = m3, cov = cov3)
w3 = np.array([-1, 1, 1])*enhance

#GAUSS 4
m4 = [-3, -3]
cov4 = covG2
g4 = multivariate_normal(mean = m4, cov = cov4)
w4 = np.array( [1, 1, 1] )*enhance

#GENERATE DATA 
#DATA OF THE FORM [f1, f2, label,  g(index), g(index2), pg3, pg4, pw1]

N = 10000

data = genData( mix[0], mix[1], m1, m2, m3, m4, cov1, cov2, cov3, cov4, w3, w4, N)
        
        
        



#ones = np.ones([N,1])     
X = data[:, 0:3]
#X = np.concatenate((ones, X), axis = 1)
Y = data[:, 3]    


#X = np.concatenate((ones, X), axis = 1) 
##Fitting SGMM
adaR = 1
alpha = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 ]
#alpha = [0.009]
n_clusters = 2
vrb = 0
cv = 10
scoring = 'neg_log_loss'
mcov = 'diag'
mx_it = 1000
mx_it2 = 8
warm = 0
km = 0
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-3),
                         max_iter = mx_it, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 1, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-2) )


#PREDICTION
Xtrain, Xtest, ytrain, ytest = model.split(X = X, y = Y, split = 0.2)

model = model.fit( Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain, kmeans = km,
                  ind2 = [0,1,2])

#PREDICT THE INTERNAL PROBABILITIES 
probTest, probTrain = model.predict_prob_int( Xtest = Xtest, Xtrain = Xtrain )

res = sgmmResults( model , probTest, probTrain, ytest, ytrain)


#AVERAGING
#SPLIT THE DATA
trans = 40
avg = 1
wit = 7
ts = 0.9


#gausDict = experiment1(X, Y.astype( int ), model, trans = trans, averaging = avg,
#                     warm = warm, warm_it = wit, kmeans = km, train_size = ts)




indg1 = np.where( data[:, 4] == 1 )[0]
indg2 = np.where( data[:, 4] == 2 )[0]  

indg3 = np.where( data[:, 5] == 3)[0]
indg4 = np.where( data[:, 5] == 4 )[0]    

ind0 =  np.where( data[:, 3] == 0 )[0] 
ind1 =  np.where( data[:, 3] == 1 )[0] 

ind0E = np.where( res['yTrain'] == 0)[0].tolist()
ind1E = np.where( res['yTrain'] == 1)[0].tolist()
ind0ET = np.where( res['yTest'] == 0)[0].tolist()
ind1ET = np.where( res['yTest'] == 1)[0].tolist()

XX = np.concatenate((Xtrain, Xtest), axis = 0)
ind1E.extend(ind1ET)
ind0E.extend(ind0ET)
fig, ax = plt.subplots(4, 1, figsize = [15, 15] )

s = 0.3

ax[0].scatter( data[indg1, 1], data[indg1, 2] , s = s )
ax[0].scatter( data[indg2, 1], data[indg2, 2] , s = s )
ax[0].legend(['Gaussian1', 'Gaussian2'])

ax[1].scatter( data[indg3, 1], data[indg3, 2] , s = s )
ax[1].scatter( data[indg4, 1], data[indg4, 2], s = s  )
ax[1].legend(['Gaussian3', 'Gaussian4'])

ax[2].scatter( data[ind1, 1], data[ind1, 2], s = s )
ax[2].scatter( data[ind0, 1], data[ind0, 2], s = s)
ax[2].legend(['Class1T', 'Class0T'])

ax[3].scatter( XX[ind1E, 1], XX[ind1E, 2], s = s)
ax[3].scatter( XX[ind0E, 1], XX[ind0E, 2], s = s)
ax[3].legend([ 'Class1E', 'Class0E'])
ax[3].set_title("TRANSEXP 4")


#columns = ['cluster', 'size', 'high_cost%','low_cost%', 
#                       'TP', 'TN', 'FP', 'FN', 
#                       'FPR', 'specificity', 'sensitivity', 'precision',
#                       'accuracy', 'balanced accuracy', 'f1', 'auc']
#testRes = gausDict['testF']
#testResPd = pd.DataFrame(testRes, columns = columns)
#index = np.arange(40)
#
#fig, ax = plt.subplots( 1, 1)
#ax.plot( index, testResPd['precision'])
#ax.plot( index, testResPd['accuracy'])
#ax.plot( index, testResPd['sensitivity'])
#ax.plot( index, testResPd['specificity'])
#ax.plot( index, testResPd['f1'])
#ax.plot( index, testResPd['auc'])
#ax.set_xlabel('folds')
#ax.set_ylabel('Performance_Metric')
#ax.legend(['precision', 'accuracy', 'sensitivity', 'specificity', 'f1', 'auc'])
#
#
