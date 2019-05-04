#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:47:57 2019

@author: george
"""

import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
from  sklearn.model_selection import train_test_split
warnings.warn = warn
#import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

np.random.seed( seed = 0)
A = np.arange(20)
Y = np.zeros(20)
Y[0:10] = 1


for i in np.arange(10):
    
    Atr, Atest, Ytr,Ytest = train_test_split(A, Y, train_size = 0.25, 
                                             stratify = Y ,
                                             )
    
    print(Atest[10])