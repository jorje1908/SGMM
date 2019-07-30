#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:03:48 2019

@author: george
"""

import numpy as np
from numpy import logaddexp
from scipy.special import logsumexp




def _log_forward( log_A, log_p_states, log_init_states, log_forw, T, K):
    
    
    for i in range(K):#initialize
        log_forw[i,0] = log_p_states[i,0] + log_init_states[i]
        
    
    work_buffer  = np.zeros(shape = [K])
    for t in range(1,T):
        for i in range(K):
            for j in range(K):
                work_buffer[j] = log_A[j,i] + log_forw[j,t-1]
            
            log_forw[i,t] = logsumexp(work_buffer) + log_p_states[i,t]
            
            
            
            
def _log_backward(log_A, log_p_states, log_backw, T, K):
    
    
    for i in range(K):
        log_backw[i,T-1] = 0
        
    
    work_buffer  = np.zeros(shape = [K])
    for t in range(T-2, -1, -1):
        for i in range(K):
            for j in range(K):
                work_buffer[j] = log_A[i,j] + log_backw[j,t+1] + \
                                                            log_p_states[j,t+1]
            
            log_backw[i,t] = logsumexp(work_buffer)  
            
            
def _log_gamas(log_forw, log_backw, log_gammas):
    
    log_gammas = log_forw + log_backw
    
    normalize = logsumexp(log_gammas, axis = 0)
    log_gammas = log_gammas - normalize
    
    return log_gammas
    
    
def _log_xis(log_A, log_p_states, log_forw, log_backw, log_xis, T, K):
    

    for t in range(T-1):
        logzero = -np.math.inf
        for i in range(K):
            for j in range(K):
                log_xis[i,j,t] = log_forw[i, t] + log_backw[j, t+1]\
                                  +log_A[i,j] + log_p_states[j,t+1] 
                
                logzero = logaddexp(logzero, log_xis[i,j,t])
                
        for i in range(K):
            for j in range(K):
                log_xis[i,j,t] = log_xis[i,j,t] - logzero
    
    
    
        
        
    