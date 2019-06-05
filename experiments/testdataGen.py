#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:12:01 2019

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
sys.path.append('../experiments')
#sys.path.append('../oldCode')

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import multivariate_normal
from supervisedGmm import SupervisedGMM
from metricsFunctions import optimalTau, calc_metrics, metrics_cluster, sgmmResults
import time
import pandas as pd
from dataGen import genData1D, bayeOpt, genData1D2, bayesO, MC, bayesOF, MCF,\
probxG


np.random.seed( seed = 0 )

#INITIALIZING THE GAUSSIAN PARAMETERS
covG = np.array( 1.5 )
covG2 = np.array( 1.5 )
enhance = 10000

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
w4 = np.array([-b4, -1])*enhance


piX = mix
covX = [covG, covG]
covY = [covG2, covG2]
meansX = [m1, m2]
meansY = [m3, m4]
w = [w3, w4]

#dat = genData1D2(piX, meansX, covX, meansY, covY, w, 100)
L  = -32
U = 32
N = 40000

start = time.time()
#bopt, point, Dx =  bayesO(piX, meansX, covX, meansY, covY, w, L, U, N)
#mCopt = MC(piX, meansX, covX, meansY, covY, w, N)

end = time.time() - start

print( "End time 1: {:.2f}".format( end ))
#print(bopt, mCopt)

start1 = time.time()

boptF = bayesOF( piX, meansX, covX, meansY, covY, w, L, U, N )


mCoptF = MCF( piX, meansX, covX, meansY, covY, w, N )
end1= time.time() - start1

print( "End time 1: {:.2f}".format( end1 ))
print( boptF, mCoptF )



data = genData1D2( piX, meansX, covX, meansY, covY, w, 10)
x = data[:,1]

print(  len(np.where( data[:,2] == 1)[0]) )

tot  =  probxG(x, piX, meansX, covX)
tot0 =  probxG(x, [0.5], [0], [covG])
tot1 =  probxG(x, [0.5], [3], [covG])