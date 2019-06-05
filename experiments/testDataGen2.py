#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:20:22 2019

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

def gaus1d( x, p,  m, cov):
    
    px = 0
    
    for i in np.arange( len(m)):
        pii = p[i]
        mi = m[i]
        covi = cov[i]
        
        pxx = gauss(x, mi, covi)
       
        px += pxx*pii
      #  print(pii, mi, covi, px)
    return px

def gauss(x, m, cov):
    
     covS = cov**2
     den1 = 1/np.sqrt( 2*np.pi*covS )
     den2 = np.exp( -(x - m)**2/( 2*covS ))
     px = den1*den2
     
     return px
    
def proby(x, m, cov, w):
    
    py = 0
    pxs = 0
    for i in np.arange( len( m ) ):
        mi = m[i]
        covi = cov[i]
        wi = w[i]
        px = gauss( x, mi, covi )
        pxs += px
        py += px*sigm( x, wi )
    
    py = py/pxs
    return py
    
    

np.random.seed( seed = 0)

p1, p2 = 0.5, 0.5
m1, m2, m3 , m4 = 0, 3, 0, 3
covG = 1.5
cov1, cov2, cov3, cov4 =covG, covG, covG, covG
w1 = np.array([0, 1])*100
w2 = np.array([-3, -1])*100

pis = [p1, p2]
meanX = [m1, m2]
meanY = [m3, m4]
covX = [cov1, cov2]
covY = [cov3, cov4]
w = [w1, w2]


x  = genXG( pis, meanX, covX, 2 )
#y = genyG(x, meanY, covY, w)
Y = genYG(x, meanY, covY, w)
px = probxG( x, pis, meanX, covX)
pyx = probyG( x, meanY, covY, w )
pxm = gaus1d(x, pis, meanX, covX)
pym = proby(x,  meanY, covY, w)
#print(x, px, pxm, pyx, pym)


N = 100
dat = genData1D( pis, meanX, covX, meanY, covY, w, N)
X, Y = dat[:, 0:2], dat[:, 2]

bayesInt = bayesOF(pis, meanX, covX, meanY, covY, w, -15, 15, 10000)
monteC = MCF(pis, meanX, covX, meanY, covY, w, 10000)

print(" BOPT: {}, MONTE: {}".format(bayesInt, monteC))



