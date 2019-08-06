#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:35:16 2019

@author: george
"""

import numpy as np



def make_supervised( states_matrix, value = 0):
    """
    takes a matrix with values 
    (in general 0 or 1) and produces
    a matrix with 1 and -infinities
    replacing the value "value" with -inf
    """
    
    dim0 = states_matrix.shape[0]
    new_mat = np.zeros_like( states_matrix )
    
    for i in np.arange( dim0 ):
        
        rowi = states_matrix[i,:]
        rowi[np.where(rowi == value)] = -np.Inf
        
        new_mat[i,:] = rowi
        
    
    return new_mat
        
    
    