#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:12:56 2019

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
probxG, probyG, genxG, genyG, genYG, sigm, genXG, gauss

from LKEXPfunc import assignLabel, assignGen, calcStats,\
logisTrain, Qfunc, QfuncAll, likelihood, LKEXP, logisTrainScikit



##################### MAKING EXPERIMENTS TO EXPLORE HOW THE LIKELIHOOD BEHAVES
##################### IN CERTAIN SETTINGS  ###################################
np.random.seed( seed = 0)



#INITIALIZE PARAMETERS
p1, p2 = 0.5, 0.5
mF, mS = 0, 3
m1, m2, m3 , m4 = mF, mS, mF, mS
covG = 1
cov1, cov2, cov3, cov4 =covG, covG, covG, covG
enhance = 4
w1 = np.array([0, 1])*enhance
w2 = np.array([-3, 1])*enhance

#put them in list mode so the can be used for data generator
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


N = 50
batches = 200
start = 0.5
enh = 50
step = 1
wone = [0,1]
wtwo = [ -3, 1 ]


data = genData1D(pis, meanX, covX, meanY, covY, w, N)

dpos, dneg = assignLabel( data )
d1, d2 = assignGen( data )

Lall, myD, Like = QfuncAll( dpos[:,1], dneg[:,1], dpos[:,2], dneg[:,2], N)
Lall1, myD1, Like1 = QfuncAll( d1[:,1], d2[:,1], d1[:,2], d2[:,2], N)

wM, pyM = logisTrain( d1[:, 1], d1[:, 2], printO = 1)
wS, pyS = logisTrainScikit(d1[:, 1], d1[:, 2])

###### START EXPERIMENT  ######################################################
#startT = time.time()
#
#
#
#L1l, L2l, L1lk, L2lk, index = LKEXP(N, batches, start, enh, step, wone,
#                                    wtwo,pis, meanX, covX, meanY, covY)
#
#end = time.time() - startT
#print("time is :{} s".format( end) )

###   END OF EXPERIMENT  #####################################################

######     PRINTING    #######################################################

#fig, ax = plt.subplots(1, 1, figsize = [12, 12] )
#ax.plot( index, L1l )
#ax.plot( index, L2l )
#ax.plot( index, L1lk )
#ax.plot( index, L2lk )
#ax.legend([ 'Q1', 'Q2', 'L1', 'L2' ])
#ax.set_title(" Model Params: m = [{} {}], sigm = [{} {}],\n\
#             w1= [{}, {}], w2 = [{}, {}], p = [{},{}]".format(mF, mS, covG,
#             covG, w1[0], w1[1], w2[0], w2[1], p1, p2))









