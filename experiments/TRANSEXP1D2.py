#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:06:14 2019

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
from dataGen import genData1D, bayesOF, MCF

np.random.seed( seed = 0 )


#INITIALIZING THE GAUSSIANS AND TH SEPARATORS
covG = np.array( 1.5 )
covG2 = np.array( 1.5 )
enhance = 1000

mix = [0.5, 0.5]
#GAUSS 1
m1 = 0
cov1 = covG
g1 = multivariate_normal(mean = m1, cov = cov1.copy())

#GAUSS 2
m2 = 3
cov2 = covG
g2 = multivariate_normal(mean = m2, cov = cov2.copy())

#GAUSS 3
m3 = 0
cov3 = covG2
g3 = multivariate_normal(mean = m3, cov = cov3.copy())

#creating the separting hyperplane based  on points p1 p2 for third gaussian
b3 = 0
w3 = np.array([-b3, 1])*enhance

#GAUSS 4
m4 = 3
cov4 = covG2
g4 = multivariate_normal(mean = m4, cov = cov4.copy())

#creating the separting hyperplane based  on points p1 p2 for third gaussian

b4 =  3
w4 = np.array([ -b4, -1])*enhance


piX = mix
covX = [covG, covG]
covY = [covG2, covG2]
meansX = [m1, m2]
meansY = [m3, m4]
w = [w3, w4]


L  = -32
U = 32
N = 4000
start1 = time.time()

boptF = bayesOF( piX, meansX, covX, meansY, covY, w, L, U, N )
mCoptF = MCF( piX, meansX, covX, meansY, covY, w, N )

end1= time.time() - start1

print( "End time 1: {:.2f}".format( end1 ))
print( boptF, mCoptF )

##Fitting SGMM
adaR = 1
#alpha = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 0.0009]
alpha = [0.00000000001]          
n_clusters = 2
vrb = 0
cv = 10
scoring = 'neg_log_loss'
mcov = 'diag'
mx_it = 1000
mx_it2 = 90
warm = 0
km = 1
mod = 1
mix1 = 0.5
#sparse means params
m_LR = 0.0001
m_sp_it1 = 1
m_sp_it2 = 1500
m_choice = 0
m_sparseL = 1
m_sparse = 0
#INITIALIZE THE MODEL
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-10),
                         max_iter = mx_it, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 0, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-2) , mix = mix1,
                         m_sparse = m_sparse, m_sparseL = m_sparseL, m_LR = m_LR,
                         m_sp_it1 = m_sp_it1, m_sp_it2 = m_sp_it2, 
                         m_choice = m_choice)

N = 50000
N1 = 20
split = 0.05
#GENERATE DATA 
data =  genData1D(piX, meansX, covX,  meansY,  covY, w, N)
    
#indTest = genData1D( mix[0], mix[1], m1, m2, m3, m4, cov1, cov2,
                                                        # cov3, cov4, w3, w4, N1)

X, Y = data[:, 0:2], data[:, 2]

Xtrain, Xtest, ytrain, ytest = model.split(X = X, y = Y, split = split)

model = model.fit( Xtrain = Xtrain, Xtest = [], ytrain = ytrain, kmeans = km,
                          ind2 = [ 1], mod = mod )

#PREDICT THE INTERNAL PROBABILITIES 
probTest, probTrain = model.predict_prob_int( Xtest = Xtest, Xtrain = Xtrain )
probTest1 = model.predict_proba( Xtest )
memb1 = model.predict_GMMS( Xtest )
res = sgmmResults( model , probTest.copy(), probTrain.copy(), ytest, ytrain, tau = None, mode = 3)

#PLOTTING
plus = np.where(Y == 1 )[0]
minus = np.where( Y == 0 )[0]

fig, ax = plt.subplots( 1, 1, figsize = (12, 12))

ax.scatter( X[plus,1], np.zeros_like( X[plus, 1]), s = 0.3)
ax.scatter( X[minus,1], np.zeros_like( X[minus, 1]), s = 0.3)
ax.legend( ['1', '0'])


