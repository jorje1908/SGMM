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
import time


def experiment1( X, Y, model, averaging = 10, train_size = 0.25, trans = 10 ):
    
    """Transduction and averaging experiment """
    
    np.random.seed( seed = 0)
    
    N = X.shape[0]
    idx = np.arange(N)
    index100 = []
    totResTrain = []
    totResTest = []
    start = time.time()
    for  i in np.arange( averaging ): #Run with "averaging Random Splits
        print("\n################################################")
        print("ITERATION: {} OF AVERAGING".format( i))
        
        #SPLIT DATA INTO TRAINING AND TEST
        Xtrain, Xtest, ytrain, ytest, itr, itst = train_test_split( X, Y, idx,
                                                       train_size = train_size,
                                                       stratify = Y,
                                                       random_state = i )
        
        
        index100.append( itr[100])
        
        #STUDY TRANSDACTION 
        #for each of the transdactional splits we need the results
        _, _,npTest, npTrain = transduction(model, Xtrain, Xtest,
                                                     ytrain, ytest,
                                                     trans = trans )
        totResTrain.append( npTrain )
        totResTest.append( npTest )
        print("################ END OF ITERATION ###########################")
    
    #DO THE AVERAGING  OUTSIDE OF LOOP
    end = time.time() - start
    print("END OF ALGORITHM TIME ELAPSED: {}s".format(end))
    testFinal = totResTest[0].copy()
    trainFinal  = totResTrain[0].copy()  

    for avg in np.arange(1, averaging):
        testFinal += totResTest[avg]
        trainFinal += trainFinal[avg]
      
    testFinal = testFinal/averaging
    trainFinal = trainFinal/averaging
    
    myDict = { 'testF': testFinal, 'trainF': trainFinal, 'testTot': totResTest,
              'trainTot': totResTrain, 'index100': index100}
    
    return myDict
        
        
        
def transduction(model,  Xtrain, Xtest, ytrain, ytest, trans = 10):
    """
    FUNCTION PERFORMING THE TRANSDACTION EXPERIMENT 
    
    """
    
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
                
        resultsTest.append(metricsTest) 
        resultsTrain.append(metricsTrain)    
        
    return resultsTest, resultsTrain, np.array(resultsTest), np.array(resultsTrain)







def AllAvg(X, Y,  train_size = 0.25, averaging = 10):
    
    """AVERAGING MACHINE LEARNING MODELS """
    
    columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']

    methods = ['L1', 'NN', 'RF', 'Ada', 'GB']
###############################################################################


    Cs = [ 0.01, 0.01, 1, 10, 100, 1000 ]
    #alpha = [0.1, 0.01, 0.001, 0.0001, 10**(-7), 1 ]
    
    idx = np.arange(X.shape[0])
    index100 = []

    start = time.time()
    
    trainRes = np.zeros([ 5,16 ])
    testRes = np.zeros([ 5,16 ])
###############################################################################
    for i in np.arange( averaging ):
        print("\n################################################")
        print("ITERATION: {} OF AVERAGING".format( i))
        
        #SPLIT DATA INTO TRAINING AND TEST
        Xtrain, Xtest, ytrain, ytest, itr, itst = train_test_split( X, Y, idx,
                                                       train_size = train_size,
                                                       stratify = Y,
                                                       random_state = i)
        
        
        index100.append( itr[100] )
        #FITTING L1 LOGISTIC REGRESSION
        pL1, probTestL1, probTrainL1 = logisticRegressionCv2( Xtrain = Xtrain,
                                                          ytrain = ytrain,
                                                          Xtest = Xtest,
                                                    ytest = ytest, Cs = Cs )
###############################################################################
        print("\n################################################")
        print("ITERATION: {} RUNNING L1 LOGISTIC REGRESSION".format( i))
     #METRICS L1 LOGISTIC REGRESSION  
        tau = optimalTau(probTrainL1, ytrain)

        metTestL1,_ = calc_metrics(custom_prob = probTestL1.copy(), tau = tau, y = ytest)
        metTrainL1 ,_ = calc_metrics(custom_prob = probTrainL1.copy(), tau = tau, y = ytrain)

        trainRes[0,:] += metTrainL1
        testRes[0,:] += metTestL1
    
    
###############################################################################
        print("\n################################################")
        print("ITERATION: {} RUNNING NEURAL NETWORKS".format( i))
        #Fitting Neural Nets
        pNN, probTestNN, probTrainNN = neural_nets( Xtrain = Xtrain,
                                                  ytrain = ytrain,
                                                  Xtest = Xtest,
                                                  ytest = ytest,
                                                  h_l_s = (4 ,4, 2))

        #Metrics Neurals Nets
        tau = optimalTau(probTrainNN, ytrain)

        metTestNN,_ = calc_metrics(custom_prob = probTestNN.copy(), tau = tau, y = ytest)
        metTrainNN ,_= calc_metrics(custom_prob = probTrainNN.copy(), tau = tau, y = ytrain)
     
        trainRes[1,:] += metTrainNN
        testRes[1,:] += metTestNN
    
###############################################################################



###############################################################################
        print("\n################################################")
        print("ITERATION: {} RUNNING RANDOM FORESTS".format( i))
        #RANDOM FORESTS
        params, probTest, probTrain = randomforests(Xtrain = Xtrain, ytrain = ytrain,
                                            Xtest = Xtest, ytest = ytest)

        tau = optimalTau(probTrain, ytrain)
        metTestRF,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, y = ytest)
        metTrainRF ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau, y = ytrain)
     
        trainRes[2,:] += metTrainRF
        testRes[2,:] += metTestRF

    


###############################################################################
        print("\n################################################")
        print("ITERATION: {} RUNNING ADA BOOST".format( i))
        #Ada boost
        params, probTest, probTrain = xboost(Xtrain = Xtrain, ytrain = ytrain,
                                             Xtest = Xtest, ytest = ytest)

        tau = optimalTau(probTrain, ytrain)
        metTestAda,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, y = ytest)
        metTrainAda ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau, y = ytrain)
     
        trainRes[3,:] += metTrainAda
        testRes[3,:] += metTestAda



###############################################################################
        print("\n################################################")
        print("ITERATION: {} RUNNING GRAD BOOST".format( i))
       #Grad boost
        params, probTest, probTrain = gradboost(Xtrain = Xtrain, ytrain = ytrain,
                                            Xtest = Xtest, ytest = ytest)

        tau = optimalTau(probTrain, ytrain)
        metTestGB,_ = calc_metrics(custom_prob = probTest.copy(), tau = tau, y = ytest)
        metTrainGB ,_= calc_metrics(custom_prob = probTrain.copy(), tau = tau, y = ytrain)

        trainRes[4,:] += metTrainGB
        testRes[4,:] += metTestGB

        end = time.time() - start
###############################################################################
    print("################################################")
    print("\n END OF AVERAGING- TIME ELAPSED: {}".format(end) )

    trainResPD = pd.DataFrame( trainRes, index = methods, columns = columns)/averaging
    testResPD = pd.DataFrame( testRes, index = methods, columns = columns)/averaging

    return trainResPD, testResPD, index100

        





