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

from sklearn.linear_model import  SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture

from LKEXPfunc import validateLogRegr


np.random.seed( seed = 0)


#INITIALIZE PARAMETERS
p1, p2 = 0.5, 0.5
mF, mS = 0, 4
m1, m2, m3 , m4 = mF, mS, mF, mS
covG = 1.0
cov1, cov2, cov3, cov4 =covG, covG, covG, covG
enhance = 3
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
N = 10000
Ntest = 1000
data =  genData1D(pis, meanX, covX,  meanY,  covY, w, N)
dataTest =  genData1D(pis, meanX, covX,  meanY,  covY, w, Ntest)

meansValidate, memb0, memb1, data0, data1 = validateLogRegr( data, sgd = 0 )

#####################################INITIALIZE MODEL #########################
adaR = 1
alpha = [0.000000000000000000000000000001]      
C = [100000000000000000000000000000000]    
n_clusters = 2
vrb = 0
cv = 10
scoring = 'neg_log_loss'
mcov = 'full'
penalty = 'l1'
mx_it = 10000
mx_it2 =  20
warm = 0
km = 1
mod = 1
mix1 = 0.0
memb_mix = 0.0
#sparse means params
m_LR = 0.0001
m_sp_it1 = 1
m_sp_it2 = 1500
m_choice = 0
m_sparseL = 1
m_sparse = 0
altern = 0
tol2 = 10**(-8)
tol = 10**(-10)
lg = 'LG'
param_grid = {'alpha': alpha}

#################    INITIALIZE THE MODEL   ##################################
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = tol,
                         max_iter = mx_it, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 0, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = tol2 , mix = mix1,
                         m_sparse = m_sparse, m_sparseL = m_sparseL, m_LR = m_LR,
                         m_sp_it1 = m_sp_it1, m_sp_it2 = m_sp_it2, 
                         m_choice = m_choice, altern = altern, C = C, 
                         log_reg = lg )

###############################################################################
#take X and Y
X, Y = data[:, 1],  data[:, 2]
Xtest, ytest = dataTest[:, 1],  dataTest[:, 2]

#FIT MODEL
simple = 1 #simple gaussians or adaptive
model = model.fit( Xtrain = np.expand_dims(X, axis = 1), Xtest = [], 
                                          ytrain = Y.astype(int), kmeans = km,
                                    ind2 = None, mod = mod , simple = simple,
                                                   comp_Lik = 0, memb_mix = memb_mix,
                                                   hard_cluster = 0)

modelG = GaussianMixture(n_components = 2, random_state = 0, init_params = 'kmeans')
modelG = modelG.fit( X.reshape(-1,1) )
memberG = modelG.predict_proba( X.reshape(-1,1) )

meanG = modelG.means_
covG = np.squeeze( modelG.covariances_ , axis = 1)
mixG = modelG.weights_

###############################################################################

#PREDICT PROBABILITIES
_, probTrain2 = model.predict_prob_int(Xtrain = X.reshape(-1,1), Xtest = Xtest.reshape(-1,1))
probTrain =  model.predict_proba( X.reshape(-1,1 ) )
probTest = model.predict_proba( Xtest.reshape(-1, 1) )



############  TAKE RESULTS ####################################################
tau = None
res =  sgmmResults( model , probTest.copy(), probTrain.copy(), ytest, Y, 
                                                  Xtest = Xtest.reshape(-1,1),
                                                          tau = None, mode = 3 )


#################### VIEW SOME PARAMETERS FOR VALIDATING ######################
view = np.concatenate(( res['memberTest'], probTest.reshape(-1,1), dataTest[:,1:]), axis = 1)
viewTrain = np.concatenate(( res['memberTr'],
                                probTrain.reshape(-1,1), data[:,1:]), axis = 1)

viewTrainPd = pd.DataFrame( view, columns = ['Memb0', 'Memb1', 'Prob1', 'Xtrain', 
                                               'Ytrain', 'Gstage1', 'Gstage2'] )

viewPd =pd.DataFrame( view, columns = ['Memb0', 'Memb1', 'Prob1', 'Xtest', 
                                               'Ytest', 'Gstage1', 'Gstage2'] )

membs = np.concatenate( ( res['memberTr'], memberG ), axis = 1)


#PRINTING RESULTS FOR UNDERSTANDING
weights = res['weights']
piss = res['pis']
means = res['means']
cov = res['cov']
accu = res['trainMet']['accuracy']
accuT = res['testMet']['accuracy']
tau = res['tau']
posP = res['posP']

w00 = np.array(weights[0] )
w11 = np.array( weights[1] )

w00 = w00/abs(w00[1])
w11 = w11/abs(w11[1])

print("\n \n############# Results ###############")
print("W0: {:.4f}, {:.4f}".format( w00[0], w00[1] ) )
print("W1: {:.4f}, {:.4f}".format( w11[0], w11[1] ) )
print("p0: {:.4f}, p1:{:.4f}".format( piss[0], piss[1] ) )
print("pG0: {:.4f}, pG1:{:.4f}".format( mixG[0], mixG[1] ) )
print("m0: {:.4f}, m1:{:.4f}".format( means[0][0], means[1][0] ) )
print("mG0: {:.4f}, mG1:{:.4f}".format( meanG[0,0], meanG[1,0] ) )
print("c0: {:.4f}, c1:{:.4f}".format( cov[0][0][0], cov[1][0][0] ) )
print("cG0: {:.4f}, cG1:{:.4f}".format( covG[0,0], covG[1,0] ) )
print("Accuracy Tr: {:.4f}".format(accu.iloc[0]))
print("Accuracy Test: {:.4f}".format(accuT.iloc[0]))
print("tau: {:.4f}".format( tau ))
print("Percentage of Positive: {:.4f}".format( posP ))

#PRINT OPTIMAL ERROR
bayesInt = bayesOF(pis, meanX, covX, meanY, covY, w, -15, 15, 10000)
monteC = MCF(pis, meanX, covX, meanY, covY, w, 10000)

print("BOPT: {}, MONTE: {}".format(bayesInt, monteC))
print("#####################################")

posIndx = np.where( Y == 1)[0]
negIndx = np.where( Y == 0)[0]

Xpos = X[ posIndx]
Xneg = X[ negIndx]

printT = 0

if printT == 1:
    fig, ax = plt.subplots(1,1, figsize = [12, 12] )
    ax.scatter( Xpos, np.zeros( Xpos.shape[0] ), s = 0.1)
    ax.scatter( Xneg, np.zeros( Xneg.shape[0] ), s = 0.1)
    ax.legend(['ones', 'zeros'])












