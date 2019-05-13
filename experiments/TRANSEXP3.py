#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:52:36 2019

@author: george
"""

import sys

sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import multivariate_normal
from supervisedGmm import SupervisedGMM
from metricsFunctions import optimalTau, calc_metrics, metrics_cluster
import time
import pandas as pd
from experFuncs import transduction, experiment1

np.random.seed( seed = 0 )


mix = [0.5, 0.5]
#GAUSS 1
m1 = [ -5, 0 ]
cov1 = np.array( [[2,0], [0,2]])
g1 = multivariate_normal(mean = m1, cov = cov1)

#GAUSS 2
m2 = [ 5, 0 ]
cov2 = np.array( [[2,0], [0,2]])
g2 = multivariate_normal(mean = m2, cov = cov2)

#GAUSS 3
m3 = [0, 3]
cov3 = np.array( [[2,0], [0,2]] )
g3 = multivariate_normal(mean = m3, cov = cov3)
w3 = np.array([0, 1])

#GAUSS 4
m4 = [0, -3]
cov4 = np.array( [[2,0], [0,2]] )
g4 = multivariate_normal(mean = m4, cov = cov4)
w4 = np.array( [0, -1] )


#GENERATE DATA 
#DATA OF THE FORM [f1, f2, label,  g(index), g(index2), pg3, pg4, pw1]

N = 10000

data = np.zeros( [N, 8] )

for point in np.arange( N ):
    
    #1 CHOOSE GAUSSIAN 1 or 2 
    indx = np.random.choice( [1, 2], size = 1, p = mix)
    
    if indx == 1:
        x = np.random.multivariate_normal( mean = m1, cov = cov1 , size = 1)
    else:
        x = np.random.multivariate_normal( mean = m2, cov = cov2 , size = 1)
        
    data[point, 0:2]  = x
    data[point, 3] = indx
    
    #choose second layer of gaussians for x
    
    p3 = g3.pdf( x )
    p4 = g4.pdf( x )
    
    data[point, 5] = p3
    data[point, 6] = p4
    
    p33 = p3/( p3 + p4 )
    p44 = 1 - p33
    
    #chose label for second layer of gaussian
    indx2 = np.random.choice( [3, 4], size = 1, p = [p33, p44])
    data[point, 4] = indx2
    
    if indx2 == 3:
        plab1 = 1/ ( 1 + np.exp( -x@w3) )
        
    else:
        plab1 = 1/ ( 1 + np.exp( -x@w4) )
        
    
    plab0 = 1 - plab1
    #print("HERE")
    y = np.random.choice( [0, 1], size = 1, p = [plab0[0], plab1[0]] )
    data[point, 2] = y
    data[point, 7] = plab1
        
        
        
indg1 = np.where( data[:, 3] == 1 )[0]
indg2 = np.where( data[:, 3] == 2 )[0]  

indg3 = np.where( data[:, 4] == 3)[0]
indg4 = np.where( data[:, 4] == 4 )[0]    

ind0 =  np.where( data[:, 2] == 0 )[0] 
ind1 =  np.where( data[:, 2] == 1 )[0] 

fig, ax = plt.subplots(3, 1, figsize = [12, 12] )

ax[0].scatter( data[indg1, 0], data[indg1, 1] )
ax[0].scatter( data[indg2, 0], data[indg2, 1] )
ax[0].legend(['Gaussian1', 'Gaussian2'])

ax[1].scatter( data[indg3, 0], data[indg3, 1] )
ax[1].scatter( data[indg4, 0], data[indg4, 1] )
ax[1].legend(['Gaussian3', 'Gaussian4'])

ax[2].scatter( data[ind1, 0], data[ind1, 1] )
ax[2].scatter( data[ind0, 0], data[ind0, 1] )
ax[2].legend(['Class1', 'Class0'])


ones = np.ones([N,1])     
X = data[:, 0:2]
X = np.concatenate((ones, X), axis = 1)
Y = data[:, 2]    


X = np.concatenate((ones, X), axis = 1) 
##Fitting SGMM
adaR = 1
alpha = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 ]
n_clusters = 4
vrb = 0
cv = 10
scoring = 'neg_log_loss'
mcov = 'diag'
mx_it = 1000
mx_it2 = 10
warm = 0
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-3),
                         max_iter = mx_it, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 1, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-2) )

#SPLIT THE DATA
trans = 40
avg = 10
wit = 7
ts = 0.3
km = 1
#Xtrain, Xtest, ytrain, ytest = model.split( X = X, y = Y, split = 0.25)
gausDict = experiment1(X, Y.astype( int ), model, trans = trans, averaging = avg,
                       warm = warm, warm_it = wit, kmeans = km, train_size = ts)









    
    
    
    
    
