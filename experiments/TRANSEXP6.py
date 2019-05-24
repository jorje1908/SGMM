#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:24:19 2019

@author: george
"""

import sys

sys.path.append('../SGMM')
sys.path.append('../metrics')
sys.path.append('../loaders')
sys.path.append('../oldCode')
sys.path.append('../visual')
sys.path.append('../testingCodes')
sys.path.append('../otherModels')
#sys.path.append('../oldCode')

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

#creating the separting hyperplane based  on points p1 p2 for third gaussian
p31 = np.array([-2, 3])
p32 = np.array([4, -3])
w3p = p32 - p31
w3v = np.array( [-w3p[1], w3p[0]])
b3 = -w3v@p31
w3 = np.array([b3, w3v[0], w3v[1]])*enhance

#GAUSS 4
m4 = [-3, -3]
cov4 = covG2
g4 = multivariate_normal(mean = m4, cov = cov4)

#creating the separting hyperplane based  on points p1 p2 for third gaussian
p41 = np.array([-4, 3])
p42 = np.array([2, -3])
w4p = p42 - p41
w4v = np.array( [-w4p[1], w4p[0]])
b4 = -w4v@p41
w4 = np.array([b4, w4v[0], w4v[1]])*enhance


#GENERATE DATA 
#DATA OF THE FORM [f1, f2, label,  g(index), g(index2), pg3, pg4, pw1]

N = 1000 #console 6 N = 1000 train _size 100, console 5 N = 1000, train_size = 50

data = genData( mix[0], mix[1], m1, m2, m3, m4, cov1, cov2, cov3, cov4, w3, w4, N)
        
        
        



#ones = np.ones([N,1])     
X = data[:, 0:3]
#X = np.concatenate((ones, X), axis = 1)
Y = data[:, 3]    


#X = np.concatenate((ones, X), axis = 1) 
##Fitting SGMM
adaR = 1
alpha = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 0.0009]
alpha = [0.000009]
n_clusters = 2
vrb = 0
cv = 10
scoring = 'neg_log_loss'
mcov = 'diag'
mx_it = 1000
mx_it2 = 80
warm = 0
km = 1
mod = 1
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-10),
                         max_iter = mx_it, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 1, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-2) )


#PREDICTION
Xtrain, Xtest, ytrain, ytest = model.split(X = X, y = Y, split = 0.1)

model = model.fit( Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain, kmeans = km,
                  ind2 = [1,2], mod = mod)

#PREDICT THE INTERNAL PROBABILITIES 
probTest, probTrain = model.predict_prob_int( Xtest = Xtest, Xtrain = Xtrain )

res = sgmmResults( model , probTest, probTrain, ytest, ytrain)


#AVERAGING
#SPLIT THE DATA
trans = 50
avg = 10
wit = 12
ts = 0.05


gausDict = experiment1(X, Y.astype( int ), model, trans = trans, averaging = avg,
                     warm = warm, warm_it = wit, kmeans = km, train_size = ts,
                     fitmod = mod)




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
ax[3].set_title("TRANSEXP 6")


#TRANSDUCTION RESULTS
columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
testRes = gausDict['testF']
testResPd = pd.DataFrame(testRes, columns = columns)
index = np.arange(len(testResPd['precision']))

fig, ax = plt.subplots( 1, 1)
#ax.plot( index, testResPd['precision'])
#ax.plot( index, testResPd['accuracy'])
#ax.plot( index, testResPd['sensitivity'])
#ax.plot( index, testResPd['specificity'])
#ax.plot( index, testResPd['f1'])
ax.plot( index, testResPd['auc'])
ax.set_xlabel('folds')
ax.set_ylabel('Performance_Metric')
ax.legend(['precision', 'accuracy', 'sensitivity', 'specificity', 'f1', 'auc'])



print('HERE RUN ONLY TRANSEXP6')