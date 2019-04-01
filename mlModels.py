#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:42:33 2019

@author: george
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, \
 balanced_accuracy_score, f1_score
#from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import roc_curve





def logisticRegressionCv2(Xtrain = None, ytrain = None, Xtest = None,
                                  ytest = None, Cs = [10], penalty = 'l1',
                                 solver = 'saga', scoring = 'f1'):
    
    model = LogisticRegressionCV(Cs = Cs, penalty = penalty, random_state = 0,
                                 solver = solver, scoring = scoring)\
                                 .fit(Xtrain, ytrain)
                                 
    probTrain = model.predict_proba( Xtrain )[:, 1]
    probTest = model.predict_proba( Xtest )[:, 1]
    
    
    params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
    return params, probTest, probTrain



def neural_nets(Xtrain = None, ytrain = None, Xtest = None,
                          ytest = None, h_l_s = (5, 3, 2), cv = 2):
    
    
     sgd  =  MLPClassifier( hidden_layer_sizes = h_l_s, early_stopping = True,
                                                              random_state = 0)
     param_grid = {'alpha' : [0.1,  0.01, 0.001, 1]}
            
     model = GridSearchCV( sgd, param_grid = param_grid, 
                                   n_jobs = -1, 
                               scoring = 'f1', cv = cv).fit(Xtrain, ytrain)
            
     probTrain = model.predict_proba( Xtrain )[:, 1]
     probTest = model.predict_proba( Xtest )[:, 1]
     
     params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
     return params, probTest, probTrain




def randomforests(Xtrain = None, ytrain = None, Xtest = None,
                          ytest = None):
    
    #param_grid = {'alpha' : [0.1,  0.01, 0.001, 1]}
    return
   # sgd = RandomForestClassifier()
    
    
    
    