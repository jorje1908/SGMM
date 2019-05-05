#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:47:42 2019

@author: george
"""

import numpy as np
import pandas as pd



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

from supervisedGmm import SupervisedGMM
from loaders2 import loader
from experFuncs import AllAvg


#SCRIPT THAT RUNS ALL THE ML MODELS AVERAGING THEM 



np.random.seed( seed = 0 )
###############################################################################

#READING DATA SETTING COLUMNS NAMES FOR METRICS
file1 = '/home/george/github/sparx/data/sparcs00.h5'
file2 = '/home/george/github/sparx/data/sparcs01.h5'
data, dataS, idx1 = loader(2000, 300, file1, file2)

cols = data.columns
#drop drgs and length of stay
colA = cols[761:1100]
colB = cols[0]
data = data.drop(colA, axis = 1)
data = data.drop(colB, axis = 1)
colss = data.columns.tolist()

tr_sz = 0.25
avg = 5
X = data.iloc[:,0:-1].values
Y = data.iloc[:,-1].values

#SPARCS DATASET

resTr, resTest, index100 = AllAvg( X, Y, train_size = tr_sz, averaging  = avg )

Directory = "Results/sparx/"
resTr.to_csv(Directory + "resTrSpall.csv", index = False, float_format = '%.3f')
resTest.to_csv(Directory + "resTestSpall.csv", index = False, float_format = '%.3f')





