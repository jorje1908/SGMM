#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 23:34:23 2019

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
import scipy as sc
from scipy.stats import multivariate_normal




def genData(p1, p2, m1, m2, m3, m4, cov1, cov2, cov3, cov4, w3, w4, N):
    """generates synthetic data in the following fashion
    1) pick a Gaussian according to mixing weights p1, p2
    2) generate a data point according to the gaussian picked
    3) find the probability of the point belonging to one of the 
    Gaussians 3, 4
    4) classify the point in one of them 
    5) use the classification Rule  of the sepcific gaussian
    6) apply a label to the point 0 or 1
    """
    
    #mixing weights, gaussians
    mix = [p1, p2]
   # g1 = multivariate_normal(mean = m1, cov = cov1)
   # g2 = multivariate_normal(mean = m2, cov = cov2)
    g3 = multivariate_normal(mean = m3, cov = cov3)
    g4 = multivariate_normal(mean = m4, cov = cov4)
    
    #initialize data
    data = np.zeros( [N, 9] )
    
    for point in np.arange( N ):
    
        #1 CHOOSE GAUSSIAN 1 or 2 
        indx = np.random.choice( [1, 2], size = 1, p = mix)
    
        if indx == 1:
            x = np.random.multivariate_normal( mean = m1, cov = cov1 , size = 1)
        else:
            x = np.random.multivariate_normal( mean = m2, cov = cov2 , size = 1)
            
        one = [1, x[0,0], x[0,1]]
       
        x = np.array( one )        
        
        data[point, 0:3]  = x
        data[point, 4] = indx
    
        #choose second layer of gaussians for x
    
        p3 = g3.pdf( x[1:] )
        p4 = g4.pdf( x[1:] )
    
        data[point, 6] = p3
        data[point, 7] = p4
    
        p33 = p3/( p3 + p4 )
        p44 = 1 - p33
    
        #chose label for second layer of gaussian
        indx2 = np.random.choice( [3, 4], size = 1, p = [p33, p44])
        data[point, 5] = indx2
       
        if indx2 == 3:
            plab1 = 1/ ( 1 + np.exp( -x@w3) )
        
        else:
            plab1 = 1/ ( 1 + np.exp( -x@w4) )
        
        plab0 = 1 - plab1
       
        y = np.random.choice( [0, 1], size = 1, p = [plab0, plab1] )
        data[point, 3] = y
        data[point, 8] = plab1
        
        #END OF LOOP FOR DATA GENERATION
        
        
    return data      

def genData1D(p1, p2, m1, m2, m3, m4, cov1, cov2, cov3, cov4, w3, w4, N):
    """generates synthetic data in the following fashion
    1) pick a Gaussian according to mixing weights p1, p2
    2) generate a data point according to the gaussian picked
    3) find the probability of the point belonging to one of the 
    Gaussians 3, 4
    4) classify the point in one of them 
    5) use the classification Rule  of the sepcific gaussian
    6) apply a label to the point 0 or 1
    """
    
    #mixing weights, gaussians
    mix = [p1, p2]
   # g1 = multivariate_normal(mean = m1, cov = cov1)
   # g2 = multivariate_normal(mean = m2, cov = cov2)
    g3 = multivariate_normal(mean = m3, cov = cov3)
    g4 = multivariate_normal(mean = m4, cov = cov4)
    
    #initialize data
    data = np.zeros( [N, 8] )
    
    for point in np.arange( N ):
    
        #1 CHOOSE GAUSSIAN 1 or 2 
        indx = np.random.choice( [1, 2], size = 1, p = mix)
    
        if indx == 1:
            x = np.random.normal( loc = m1, scale = cov1 , size = 1)
        else:
            x = np.random.normal( loc = m2, scale = cov2 , size = 1)
            
        one = [1, x[0] ]
       
        x = np.array( one )        
        
        data[point, 0:2]  = x
        data[point, 3] = indx
    
        #choose second layer of gaussians for x
    
        p3 = g3.pdf( x[1:] )
        p4 = g4.pdf( x[1:] )
    
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
       
        y = np.random.choice( [0, 1], size = 1, p = [plab0, plab1] )
        data[point, 2] = y
        data[point, 7] = plab1
        
        #END OF LOOP FOR DATA GENERATION
        
        
    return data          
        