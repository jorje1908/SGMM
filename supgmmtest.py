#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:55:20 2019

@author: george
"""

import numpy as np
import pandas as pd



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from supervisedGmm import SupervisedGMM
from metricsFunctions import calc_metrics, CalculateSoftLogReg, optimalTau
from superGmmMother import superGmmMother
from loaders2 import loader


np.random.seed( seed = 0)
file1 = '/home/george/github/sparx/data/sparcs00.h5'
file2 = '/home/george/github/sparx/data/sparcs01.h5'
data, dataS, idx = loader(4000, 300, file1, file2)


mother = superGmmMother( data , n_clusters = 3)

results = mother.fit_results(fitted = 1)
#model = mother.model
#params = model.fit(Xtrain =mother.Xtrain,Xtest =  mother.Xtest,
#                   ytrain = mother.ytrain)


#gmm1 = params['Gmms'][0]
#xs = gmm1.cdf(mother.Xtrain[0])