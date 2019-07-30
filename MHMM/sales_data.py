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


from HMMs import MHMM
from HMMs_1 import MHMM as MH

np.random.seed( seed = 50 )

sales = pd.read_csv("/home/george/github/sparx/code/data/sales/sales.csv")
salesNorm = sales.iloc[:, 55:].values
salesNorm3d = np.expand_dims( salesNorm, axis = 2)


#initialize MHMM

model = MHMM(n_HMMS = 1, n_states = 2, EM_iter = 10, tol = 10**(-5))
m1 = MH(n_HMMS = 1, n_states = 2, EM_iter = 10, tol = 10**(-5))
start = time.time()
model = model.fit( data = salesNorm3d[0:3] )
print('\n \n \n Print no logs  \n \n'.format())
m1 = m1.fit(data = salesNorm3d[0:3] )
end = time.time() - start
logLi = model.logLikehood
logLi1 = m1.logLikehood

print("time elapsed: {:.4}".format(end))

#hmms
params = model.get_params()
params1 = m1.get_params()   

"""
if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', sort='time')
    
"""