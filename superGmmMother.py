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
    
    
       def __init__(self, data = None,  max_iter = 100, cv = 2, mix = 0.5,
                    Xtrain = None, Xtest = None, ytrain = None, ytest = None,
                    Cs = [10], split_size = 0.2,
                    max_iter2 = 10, penalty = 'l1', scoring = 'f1',
                    solver = 'saga', n_clusters = 2, tol = 10**(-3 ) ):
       
           
           """ 
           Initializing the supergmmMother class
           data: A pandas Matrix having as a last column the classification
                 labels. You can leave this at None value and specify directly
                 Train and Test sets
           
           max_iter : Maximum Iterations for L1 Logistic Regression to Run
                      Default: 100
           cv: Cross Validation splits, default 2
           mix: algorithm specific feature for supervisedGmm
           In case you do not specify the data variable data, you must specify
           Xtrain: Training data
           Xtest: Testing data
           ytrain: training labels
           ytest: testing labels
           Cs: A list specifying the inverse of regularization parameter,
               Default: Cs = [10]
           split_size: in case you specify data the percentage of the split
                       default 0.2 --> 20% test data 80% training data
           
           max_iter2: specific parameter to the supervised Gmm algorithm
                      iterations for EM algorithm to converge: Default 10
                      (this can have numerical Instabilities)
           penalty: Regularization type, Default L1
           scoring: Score to optimize, default F1
           solver: Regression solver, Default : 'saga' From Scikit Learn
           n_clusters : number of clusters to Use: default 2
           tol: tolerance supervisedGmm class algorithm specific
           """
           
           #Initialize the SupervisedGmm Model with parameteres destined
           #for it
           self.model = SupervisedGMM(max_iter = max_iter, cv = cv,
                                   mix = mix, Cs = Cs, max_iter2 = max_iter2,
                                   penalty = penalty, scoring = scoring, 
                                   solver = solver, n_clusters = n_clusters,
                                   tol = tol )
           
           #Keep track of the indexes after split
           self.idx1 = []
           self.idx2 = []
           
           
           if data is None: #if data is not defined in class 
               self.Xtrain = Xtrain
               self.Xtest = Xtest
               self.ytrain = ytrain
               self.ytest = ytest
               
           else: #if data are defined do the splits and assign the datasets 
               self.columns = data.columns.tolist()
               self.data = data.iloc[:,:-1].values   #last column is target 
               self.target = data.iloc[:,-1].values
               mats = self.train_test_split0()
               self.data = []  #delete data for the extra space
               self.Xtrain = mats[0]
               self.Xtest = mats[1]
               self.ytrain = mats[2]
               self.ytest = mats[3]
            
           #A dictionary containing parameteres after the  model has been fitted 
           self.fitParams  = []
       
       
       
       
       def train_test_split0(self, split_size):
            """ splits data to training and testing data"""
            
            idx = np.arange( self.data.shape[0]) #test train indexes
            Xtrain, Xtest, ytrain, ytest, idx1, idx2 = train_test_split(self.data,
                                                  self.target, idx,
                                                  test_size = split_size,
                                                  random_state = 1512)
            self.idx1 = idx1
            self.idx2 = idx2
            
            return Xtrain, Xtest, ytrain, ytest
        
        
       
       def fit_results( self, fitted = 0 ):
          #FIT SUPERVISED GAUSSIAN MODEL 
          #PRODUCE RESULTS
          
          #FITTING THE SUPERVISED MODEL 
          if fitted == 0: #if the model is not already fitted,fit it
              self.model = self.model.fit(Xtrain = self.Xtrain,
                                              Xtest = self.Xtest, 
                                              ytrain = self.ytrain )
          
          #take parameteres from the fitted model 
          self.fitParams = self.model.fitParams  
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
          metTr, metTest, r1, r2, tau = CalculateSoftLogReg( models = LogRegr,
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
                 'clustTr': clustTr, 'clustTest': clustTest, 'tau':tau}
            
          return params
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       