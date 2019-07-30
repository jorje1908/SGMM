#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:50:20 2019

@author: george
"""

import sys
sys.path.append('../data/sales')
sys.path.append('..')
#sys.path.append('/home/george/github/sparx/code/data/sales')
import numpy as np
import pandas as pd
import time


from MHMM import  HMMs

np.random.seed( seed = 0 )

sales = pd.read_csv("/home/george/github/sparx/code/data/sales/sales.csv")
salesNorm = sales.iloc[:, 55:].values
salesNorm3d = np.expand_dims( salesNorm, axis = 2)


#initialize MHMM

model = HMMs.MHMM( )
start = time.time()
model = model.fit( data = salesNorm3d[0:7])
end = time.time() - start
logLi = model.logLikehood

print("time elapsed: {:.4}".format(end))

#hmms
hmms = model.HMMS
gamma = hmms[0].gamas(salesNorm[2])
xisSum = hmms[0].sliced( salesNorm[2]).sum( axis = 1)