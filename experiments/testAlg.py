#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:28:58 2019

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

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import multivariate_normal
from supervisedGmm import SupervisedGMM
from metricsFunctions import optimalTau, calc_metrics, metrics_cluster, sgmmResults
import time
import pandas as pd
from dataGen import   genData1D, bayesO, MC, bayesOF, MCF,\
probxG, probyG, genxG, genyG, genYG, sigm, genXG


np.random.seed( seed = 0)


#INITIALIZE PARAMETERS
p1, p2 = 0.5, 0.5
mF, mS = 0, 4
m1, m2, m3 , m4 = mF, mS, mF, mS
covG = 1.5
cov1, cov2, cov3, cov4 =covG, covG, covG, covG
enhance = 10
w1 = np.array([0, 1])*enhance
w2 = np.array([-4, 1])*enhance

pis = [p1, p2]
meanX = [m1, m2]
meanY = [m3, m4]
covX = [cov1, cov2]
covY = [cov3, cov4]
w = [w1, w2]

#PRINT OPTIMAL ERROR
bayesInt = bayesOF(pis, meanX, covX, meanY, covY, w, -15, 15, 50000)
monteC = MCF(pis, meanX, covX, meanY, covY, w, 10000)

print(" BOPT: {}, MONTE: {}".format(bayesInt, monteC))

#GENERATE DATA
N = 5000
data =  genData1D(pis, meanX, covX,  meanY,  covY, w, N)
dataTest =  genData1D(pis, meanX, covX,  meanY,  covY, w, 1000)

#INITIALIZE MODEL
adaR = 1
alpha = [0.0000000000000001]          
n_clusters = 2
vrb = 1
cv = 10
scoring = 'neg_log_loss'
mcov = 'full'
mx_it = 1000
mx_it2 = 10
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
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-17),
                         max_iter = mx_it, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 0, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-2) , mix = mix1,
                         m_sparse = m_sparse, m_sparseL = m_sparseL, m_LR = m_LR,
                         m_sp_it1 = m_sp_it1, m_sp_it2 = m_sp_it2, 
                         m_choice = m_choice, altern = 1)


#take X and Y
X, Y = data[:, 0:2],  data[:, 2]
Xtest, ytest = dataTest[:, 0:2],  dataTest[:, 2]


#split data
split = 0.5

Xtrain, _, ytrain, _ = model.split( X = X, y = Y, split = split )

#FIT MODEL
simple = 0 #simple gaussians or adaptive
model = model.fit( Xtrain = Xtrain, Xtest = [], ytrain = ytrain, kmeans = km,
                          ind2 = [ 1 ], mod = mod , simple = simple)


#PREDICT PROBABILITIES
probTest, probTrain = model.predict_prob_int(Xtest = Xtest, Xtrain = Xtrain)


#TAKE RESULTS
res = sgmmResults(model , probTest.copy(), probTrain.copy(), ytest, ytrain, 
                                                          tau = None, mode = 3)


#PRINTING RESULTS FOR UNDERSTANDING
weights = res['weights']
pis = res['pis']
means = res['means']
cov = res['cov']
accu = res['trainMet']['accuracy']
accuT = res['testMet']['accuracy']
tau = res['tau']
posP = res['posP']

w0 = np.array(weights[0] )
w1 = np.array( weights[1] )

w0 = w0/abs(w0[1])
w1 = w1/abs(w1[1])

print("\n \nResults ############################")
print("W0: {:.4f}, {:.4f}".format( w0[0], w0[1] ) )
print("W1: {:.4f}, {:.4f}".format( w1[0], w1[1] ) )
print("p0: {:.4f}, p1:{:.4f}".format( pis[0], pis[1] ) )
print("m0: {:.4f}, m1:{:.4f}".format( means[0][0], means[1][0] ) )
print("c0: {:.4f}, c1:{:.4f}".format( cov[0][0][0], cov[1][0][0] ) )
print("Accuracy Tr: {:.4f}".format(accu.iloc[0]))
print("Accuracy Test: {:.4f}".format(accuT.iloc[0]))
print("tau: {:.4f}".format( tau ))
print("Percentage of Positive: {:.4f}".format( posP ))

#PRINT OPTIMAL ERROR
bayesInt = bayesOF(pis, meanX, covX, meanY, covY, w, -15, 15, 10000)
monteC = MCF(pis, meanX, covX, meanY, covY, w, 10000)

print("BOPT: {}, MONTE: {}".format(bayesInt, monteC))


posIndx = np.where( ytrain == 1)[0]
negIndx = np.where( ytrain == 0)[0]

Xpos = Xtrain[ posIndx, 1]
Xneg = Xtrain[ negIndx, 1]

fig, ax = plt.subplots(1,1, figsize = [12, 12])
ax.scatter( Xpos, np.zeros( Xpos.shape[0] ), s = 0.9)
ax.scatter( Xneg, np.zeros( Xneg.shape[0] ), s = 0.9)
ax.legend(['ones', 'zeros'])












