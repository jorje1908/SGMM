#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:32:39 2019

@author: george
"""

import numpy as np
import pandas as pd
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#import matplotlib.pyplot as plt


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
#from sklearn.cluster import KMeans
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import LogisticRegressionCV
#from sklearn.metrics import precision_score, accuracy_score, recall_score, \
# balanced_accuracy_score, f1_score
#from sklearn.mixture import GaussianMixture
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import roc_auc_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClas
#from  scipy.stats import multivariate_normal

from supervisedGmm import SupervisedGMM
from metricsFunctions import calc_metrics, CalculateSoftLogReg, optimalTau,\
                                                            logistic_cluster
        
                                


#MOTHER CLASS FOR SupervisedGMM CLASS, CALLS SupervisedGMM, FITs AND PRODUCE
#METRICS AND RESULTS

#THIS CODE HAS NUMERICAL ISSUES ON THE SPARCS DATASET
class superGmmMother():
    
    
       def __init__(self, data,  max_iter = 100, cv = 2, mix = 0.5, Cs = [10], 
                 max_iter2 = 10, penalty = 'l1', scoring = 'f1',
                 solver = 'saga', n_clusters = 2, tol = 10**(-3 ) ):
       
       
           self.model = SupervisedGMM(max_iter = max_iter, cv = cv,
                                   mix = mix, Cs = Cs, max_iter2 = max_iter2,
                                   penalty = penalty, scoring = scoring, 
                                   solver = solver, n_clusters = n_clusters,
                                   tol = tol )
           self.idx1 = []
           self.idx2 = []
           self.columns = data.columns.tolist()
           self.data = data.iloc[:,:-1].values   #last column is target 
           self.target = data['Target'].values
           mats = self.train_test_split0()
           self.data = []  #delete data for the extra space
           self.Xtrain = mats[0]
           self.Xtest = mats[1]
           self.ytrain = mats[2]
           self.ytest = mats[3]
           self.fitParams  = []
       
       
       
       
       def train_test_split0(self):
            """ splits data to training and testing data"""
            
            idx = np.arange( self.data.shape[0]) #test train indexes
            Xtrain, Xtest, ytrain, ytest, idx1, idx2 = train_test_split(self.data,
                                                  self.target, idx,
                                                  test_size = 0.2,
                                                  random_state = 1512)
            self.idx1 = idx1
            self.idx2 = idx2
            
            return Xtrain, Xtest, ytrain, ytest
        
        
       
       def fit_results( self, fitted = 0 ):
          #FIT SUPERVISED GAUSSIAN MODEL 
          #PRODUCE RESULTS
          
          #FITTING THE SUPERVISED MODEL 
          if fitted == 0:
              self.fitParams = self.model.fit(Xtrain = self.Xtrain,
                                              Xtest = self.Xtest, 
                                              ytrain = self.ytrain )
          
          model  = self.model
          #TAKING THE MODELS AND MODEL PARAMETERS
          #NOT NEEDED FOR THIS SNIPPET OF CODE (Gmms, mixes)
          #because I already have the memberships
          Gmms = model.Gmms
          mixes = model.mixes
          LogRegr = model.LogRegr
          
          #TAKING THE MEMBERSHIP FUNCTIONS
          #fitParams = {'mTrain' : mTrain, 'mTest': mTest, 'labTest': testlabels,
                     #'labTrain' : trainlabels }
                     
          mTrain = self.fitParams['mTrain' ]
          mTest =  self.fitParams[ 'mTest' ]
          
          #THE FOLLOWING LABELS ARE FOR HARD CLUSTERS
          labTest = self.fitParams[ 'labTest' ]
          labTrain = self.fitParams[ 'labTrain' ]
          
          #n_clusters = mTrain.shape[1] #NUMBER OF CLUSTERS
          
          #CALCULATE METRICS  OF SOFT CLUSTERS
          #HOLISTIC MEASURES FOR THE ALGORITHMS PERFORMANCE
          metTr, metTest, r1, r2 = CalculateSoftLogReg( models = LogRegr,
                                               Xtrain = self.Xtrain,
                                               Xtest = self.Xtest, 
                                               ytrain =  self.ytrain, 
                                               ytest = self.ytest,
                                               membTrain = mTrain,
                                               membTest = mTest )
                                               
                 
          #print(labTest)
          #return                              
          #PARAMETERS TO RETURN
          # MODEL IS OUR MODEL OBJECT WHICH INCLUDES
          #GMMS AND EVERY PARAMATER FOUND IN SuperVisedGMM class
          #metTr: metrics for training
          #metTest: metrics for testing
          #fitParams: Parameters returned by model.fit
          
          #PERFORM HARD CLUSTERING AND EXTRACT THE METRICS FPR EACH CLUSTER
          clustTr, clustTest, _, _ = logistic_cluster(Xtrain = self.Xtrain,
                                                            Xtest = self.Xtest,
                                                          ytrain = self.ytrain,
                                                            ytest = self.ytest,
                                                n_clusters = model._n_clusters,
                                                             labels = labTrain,
                                                          testlabels = labTest)
          
         
                                              
          params = {'metTr': metTr, 'metTest': metTest, 
                 'fitParams': self.fitParams, 'model': self.model,
                 'clustTr': clustTr, 'clustTest': clustTest}
            
          return params
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       