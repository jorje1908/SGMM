#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:01:34 2019

@author: george
"""
import sys

sys.path.append('..')
sys.path.append('../SGMM')
sys.path.append('../metrics')
#sys.path.append('../loaders')
sys.path.append('../oldCode')
sys.path.append('../visual')
sys.path.append('../testingCodes')
sys.path.append('../otherModels')

import numpy as np
import pandas as pd


#SAME AS LOADERS  BUT  READS THE DATA FROM ANOTHER FILE
def loader(big, small, file1, file2):
    
    filename1 = file1 
    filename2 =  file2
    
    store = pd.HDFStore(file1) 
    nrows = store.get_storer('part1').nrows
    #print(nrows)

    randomIndexes = np.random.choice( np.arange(0, nrows), size = big, replace = False )
    normalIndexes = np.arange(big)
    store.close()

    data0 = pd.read_hdf(filename1, key = 'part1', where = pd.Index(randomIndexes))
    data1 = pd.read_hdf(filename2, key = 'part2', where = pd.Index(randomIndexes))
    
#    data0 = pd.read_hdf(filename1, key = 'part1', where = pd.Index(normalIndexes))
#    data1 = pd.read_hdf(filename2, key = 'part2', where = pd.Index(normalIndexes))
    

    data = pd.concat([data0,data1], axis = 1).reset_index(drop = True)
    del data0, data1
    
    indexSma = np.random.choice(np.arange(big), size = small, replace = False)
    dataS = data.iloc[indexSma, :].reset_index(drop = True)
    
    return data, dataS, randomIndexes


def loader1(big, small):
    
    filename1 =  'sparcsR0.h5'
    filename2 =  'sparcsR1.h5'
    filename3 =  'sparcsR3.h5'
    
    store = pd.HDFStore('sparcsR0.h5') 
    nrows = store.get_storer('dataR').nrows

    randomIndexes = np.random.choice(np.arange(0, nrows), size = big, replace = False)
    store.close()

    data0 = pd.read_hdf(filename1, key = 'dataR', where = pd.Index(randomIndexes))
    data1 = pd.read_hdf(filename2, key = 'dataR1', where = pd.Index(randomIndexes))
    data2 = pd.read_hdf(filename3, key = 'dataR2', where = pd.Index(randomIndexes))
    

    data = pd.concat([data0,data1, data2], axis = 1).reset_index(drop = True)
    del data0, data1, data2
    
    indexSma = np.random.choice(np.arange(big), size = small, replace = False)
    dataS = data.iloc[indexSma, :].reset_index(drop = True)
    
    return data, dataS