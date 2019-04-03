#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:42:33 2019

@author: george
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans







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
                          ytest = None, h_l_s = (5, 3, 2), cv = 2,
                          scoring = 'f1'):
    
    
     sgd  =  MLPClassifier( hidden_layer_sizes = h_l_s, early_stopping = True,
                                                              random_state = 0)
     param_grid = {'alpha' : [0.1,  0.01, 0.001, 1, 2, 5, 10, 12, 100]}
            
     model = GridSearchCV( sgd, param_grid = param_grid, 
                                   n_jobs = -1, 
                               scoring = scoring, cv = cv).fit(Xtrain, ytrain)
            
     probTrain = model.predict_proba( Xtrain )[:, 1]
     probTest = model.predict_proba( Xtest )[:, 1]
     
     params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
     return params, probTest, probTrain


def kmeansLogRegr( Xtrain = None, ytrain = None, Xtest = None,
                  ytest = None, Cs = [10], penalty = 'l1', 
                  solver = 'saga', scoring = 'f1', n_clusters = 2):
    
    #CLUSTER WITH KMEANS
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0).\
             fit( np.concatenate(( Xtrain, Xtest ), axis = 0) )
             
    #TAKE THE LABELS
    labelsTrain = kmeans.labels_[0: Xtrain.shape[0]]
    labelsTest = kmeans.labels_[ Xtrain.shape[0]:]
    
    #TRAIN LOGISTIC REGRESSION
    models = [] 
    probTrain = []
    probTest = []
    for i in np.arange( n_clusters ):
        indxTr = np.where(labelsTrain == i)[0]
        indxTest = np.where( labelsTest == i)[0]
        params, _, _ = logisticRegressionCv2(Xtrain = Xtrain[indxTr], 
                                       ytrain = ytrain[indxTr], 
                                       ytest = ytest[indxTest],
                                       Xtest = Xtest[indxTest], 
                                       Cs = Cs, penalty = penalty,
                                       solver = solver, scoring = scoring)
        
        models.append( params['model'] ) 
        probTrain.append( params['probTrain'] )
        probTest.append( params['probTest'] )
    
    params = {'models': models,'labelsTrain': labelsTrain,
              'labelsTest': labelsTest, 'probTrain': probTrain,
              'probTest': probTest}
    
    return params
        
    
    

def randomforests(Xtrain = None, ytrain = None, Xtest = None,
                          ytest = None, cv = 2, scoring = 'f1'):
    
    "RANDOM FOREST CLASSIFIER"
    
    param_grid = {'n_estimators' : [20,  40, 60, 80, 100, 120, 150] }
    forest  = RandomForestClassifier()
    
    model = GridSearchCV( forest, param_grid = param_grid, 
                                  n_jobs = -1, 
                                  scoring = scoring, cv = cv).\
                                  fit(Xtrain, ytrain) #fit model 
    
    
    probTrain = model.predict_proba( Xtrain )[:, 1]
    probTest = model.predict_proba( Xtest )[:, 1]
     
    params = {'model': model, 'probTrain': probTrain, 'probTest': probTest}
    
    return params, probTest, probTrain
   
    
    
    
    