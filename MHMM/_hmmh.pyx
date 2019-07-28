#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:55:00 2019

@author: george
"""

#cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np







def _forward( double[:,:] A,  double[:,:] p_states,
              double[:] init_states,  double[:,:] forw,
              int T,  int K):
    
    """
    
    Cython implementation of forward algorithm
    
    """
    
    cdef int t,i,j = 0
    
    with nogil:
    
        for i in range(K):
            forw[i,0] = init_states[i]*p_states[i,0]
    
        for t in range(1, T):
                for i in range(K):
                    for j in range(K):
                        forw[i,t] +=  A[j,i]*forw[j,t-1]
                   
                    forw[i,t] *= p_states[i,t]
                    
                    
                    
def _backward( double[:,:] A,  double[:,:] p_states,
                   double[:] init_states,  double[:,:] backw,
                                              int T,  int K):
    
    
    cdef int t,i,j = 0
    
    with nogil:
        
        for i in range(K):
            backw[i,T-1] = 1
        
        
        for t in range( T-2, -1, -1):
            for i in range(K):
                for j in range(K):
                    backw[i,t] += A[i,j]*backw[j,t+1]*p_states[j,t+1]
    
    
    
    
    