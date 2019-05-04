#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:12:34 2019

@author: george
 
"""
import numpy as np
import pandas as pd



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

from  sklearn.model_selection import train_test_split
from supervisedGmm import SupervisedGMM
from metricsFunctions import calc_metrics, metrics_cluster, optimalTau, predict_y
#from superGmmMother import superGmmMother
from loaders2 import loader
from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
kmeansLogRegr, xboost, gradboost


def experiment1( X, Y, model, averaging = 10, train_size = 0.25, trans = 10,
                trans_yes = 0):
    
    """Transduction and averaging experiment """
    
    np.random.seed( seed = 0)
    
    N = data.shape[0]
#    Ntrain = int( N * train_split )
    idx = np.arange(N)
    index100 = []
    totalResults = []
    for  i in np.arange( averaging ): #Run with "averaging Random Splits
        
        #SPLIT DATA INTO TRAINING AND TEST
        Xtrain, Xtest, ytrain, ytest, itr, itst = train_test_split( X, Y, idx,
                                                       train_size = train_size,
                                                       stratify = Y)
        
        #test Size
        testSize = ytest.shape[0]
        indexTest = np.arange( testSize )
        
        
        index100.append( itr[100])
        
        if trans_yes == 1:
            #STUDY TRANSDACTION 
            #for each of the transdactional splits we need the results
            results = []
            for k in np.arange(trans): #FOR EACH OF THE TRANSDACTION SPLITS
                
                # set batch size 
                batch_size = int(testSize/( k + 1 ))
                predictionsT = np.zeros( testSize )
                
                for l in np.arange( k+1 ):  #FOR EACH OF THE BATCHES
                    begin = l*batch_size
                    end = l*batch_size + batch_size
                    
                    if l == k:
                        model = model.fit( Xtrain = Xtrain,
                                      Xtest = Xtest[ begin:],
                                      ytrain = ytrain)
                        ind = indexTest[begin:]
                        
                    else:
                        model = model.fit( Xtrain = Xtrain,
                                      Xtest = Xtest[ begin:end ],
                                      ytrain = ytrain)
                        ind = indexTest[begin:end]
                    
                    #TAKE THE PROBABILITIES OF TRAIN AND TEST
                    probTest, probTrain =  model.predict_prob_int(
                                                    Xtest = Xtest[ind],
                                                      Xtrain = Xtrain )
                    #take the optimal tau
                    tau = optimalTau(probTrain, ytrain)
                    
                    #take the batch prediction  for test batch
                    predTest = predict_y( probTest, tau)
                    #add the batch predictions to the whole prediction matrix
                    predictionsT[ind] = predTest
                    #take training predictions for last batch
                    if l == k:
                        predTrain = predict_y( probTrain, tau)
                    
                    #END OF BATCH TRAINING
                    
                metricsTest = calc_metrics( custom_prob = predTest)
                metricsTrain = calc_metrics( custom_prob = predTrain)
                
                results.append([metricsTest, metricsTrain])
                #END OF TRANSDACTION TESTING
            #for each of the averaging methods take the results    
            totalResults.append(results)
                
            
            
        
    return totalResults, index100
        
        
        
def transduction(model,  Xtrain, Xtest, ytrain, ytest, trans = 10):
    """
    FUNCTION PERFORMING THE TRANSDACTION EXPERIMENT 
    
    """
    results = []
    testSize = Xtest.shape[0]
    indexTest = np.arange( testSize )
    resultsTest = []
    resultsTrain = []
    for k in np.arange( trans ): #FOR EACH OF THE TRANSDACTION SPLITS
                
        # set batch size 
        batch_size = int(testSize/( k + 1 ))
        predictionsT = np.zeros( testSize )
                
        for l in np.arange( k + 1 ):  #FOR EACH OF THE BATCHES
            begin = l*batch_size
            end = l*batch_size + batch_size
            
            print("Transdaction Iteration {}, batch: {}".format(k,
                                                              batch_size))
            
            print("Total Test Size: {}, begin: {}, end: {}".format(
                                                        testSize, begin, end))
                    
            if l == k:
                model = model.fit( Xtrain = Xtrain,
                                      Xtest = Xtest[ begin:],
                                      ytrain = ytrain)
                ind = indexTest[begin:]
                        
            else:
                model = model.fit( Xtrain = Xtrain,
                                      Xtest = Xtest[ begin:end ],
                                      ytrain = ytrain)
                ind = indexTest[begin:end]
                    
                #TAKE THE PROBABILITIES OF TRAIN AND TEST
            probTest, probTrain =  model.predict_prob_int(
                                                    Xtest = Xtest[ind],
                                                      Xtrain = Xtrain )
            #take the optimal tau
            tau = optimalTau(probTrain, ytrain)
                    
            #take the batch prediction  for test batch
            predTest = predict_y( probTest, tau)
            #add the batch predictions to the whole prediction matrix
            predictionsT[ind] = predTest
            #take training predictions for last batch
            if l == k:
                predTrain = predict_y( probTrain, tau)
                    
            #END OF BATCH TRAINING
                    
        metricsTest,_ = calc_metrics( custom_prob = predictionsT, y = ytest)
        metricsTrain,_ = calc_metrics( custom_prob = predTrain, y = ytrain)
                
        resultsTest.append([metricsTest]) 
        resultsTrain.append([metricsTrain])    
        
    return resultsTest, resultsTrain
        





