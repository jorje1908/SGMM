#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:36:54 2019

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

#from sklearn.model_selection import train_test_split
#from sklearn.cluster import KMeans
#from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, \
 balanced_accuracy_score, f1_score
#from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve






def CalculateSoftLogReg( models = [],  Xtrain = [], Xtest = [], ytrain = [],
                        ytest = [],  membTrain = [], membTest = []):
                         
            """
                Calculates The soft clustering metrics
                
            """
            #NUMBER OF CLUSTERS  
            clusters = membTrain.shape[ 1 ]
                
            #INITIALIZE THE PROBABILITY MATRICES
            #FOR TRAIN AND TEST DATA
            probTest = np.zeros( [Xtest.shape[0] ] )
            probTrain = np.zeros( [Xtrain.shape[0] ] )
            
            for i in np.arange( clusters ): #FOR EACHH CLUSTER
                model = models[i]
                
                #PROBABILITY OF EACH POINT TO BELONG IN CLASS 1
                #TO USE FOR THRESHOLDING
                # FOR TRAIN AND TEST DATA
                probTest = probTest + model.predict_proba( Xtest )[ :, 1] \
                                                        * membTest[ :, i ]
                probTrain = probTrain + model.predict_proba( Xtrain )[ :, 1] \
                                                        * membTrain[ :, i ]
                
            
            #ALL THE METRICS WE WILL CALCULATE
            columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
            
            #CALCULATE THE OPTIMAL THRESHOLD THAT MAXIMIZES THE F1
            #MEASURE
            
            
            tau = optimalTau( probTrain, ytrain)  #calculate optimal 
                                                  #tau on the training data
                
            #CALCULATING THE METRICS FOR TEST AND THE METRICS FOR TRAIN
            metricsTrain, roc1 =  calc_metrics(y = ytrain, tau = tau,
                                               custom_prob = probTrain)
            metricsTest, roc2 =   calc_metrics( y = ytest, tau = tau,
                                               custom_prob = probTest)
            
            #MAKING THE METRICS INTO DATAFRAMES WITH THEIR RESPECTIVE COLUMNS
            
            metricsTrain = pd.DataFrame( [metricsTrain], columns = columns)
            metricsTest = pd.DataFrame( [metricsTest], columns = columns)
            
            return metricsTrain, metricsTest, roc1, roc2, tau
        
        
def optimalTau(probabilities, ylabels):
            
            """ Finds the Optimal tau based on the F1 score
                Input: Probabilities of train data of being class 1,
                ylabels of training data"""
            
            #STEP 1 SORT PROBABILITIES AND LABELS
            sortedIndexes = np.argsort( probabilities )
            probabilities1 = probabilities[ sortedIndexes ]
            ylabels1 = ylabels[ sortedIndexes ]
            
            #INITIALIZE THRESHOLD TO BE 0
            #SO EVERY POINT  IS PREDICTED AS CLASS 1
            
           # initialPrediction = np.ones( probabilities1.shape[0] ) #matrix with all 1's - INITIAL PREDICTION
            
            TP = len( np.where( ylabels1 == 1)[0] )  #AT THE BEGGINING THE TRUE POSITIVES ARE THE SAME 
                                                    #AS THE POSITIVE LABELS OF THE DATASET
            
            FN = 0 #AT THE BEGGINING  WE HAVE 0 POSITIVE POINTS  CLASSIFIED AS NEGATIVE
            #XIAO HERE YOU WILL PUT  ylabels == -1
            FP = len( np.where( ylabels1 == 0)[0] )
            
            precision = TP/(TP + FP)
            recall = TP/ (TP + FN)
            
#            print(precision, recall, TP, FN, FP)
#            return
            f1 = ( 2*precision*recall )/( precision + recall )   
            
            threshold = 0
            prob_F1 = [[threshold, f1]]
            
            for i, probability in enumerate( probabilities1 ):
                
               # print( " Iteration: {}".format(i))
                
                
                if ylabels1[i] == 1:
                    
                    TP -= 1
                    FN += 1
                
                if ylabels1[i] == 0: #FOR XIAO HERE -1
                    FP -= 1
                    
                if (TP + FP == 0):
                    
                    precision = 0
                    
                else:
                    precision = TP/(TP + FP)
                    
                recall = TP/ (TP + FN)
                
                if (precision + recall) == 0:
                
                    f1new = 0
                    
                else:
                    
                    f1new = ( 2*precision*recall )/( precision + recall )  
                
                prob_F1.append( [probability, f1new] )   #thresholds with F1 scores if you want to draw a graph
                
                if f1new >= f1 :
                    threshold = probability
                    f1 = f1new
                    prec = precision  #you can return precision
                    rec = recall      #you can return  recall
                                      # these are mostly for correctness
                                      #checking
            
            return threshold #, f1, np.array(prob_F1), prec, rec
        
def calc_metrics(model = [], cluster = -1, y = [], tau = 0.5, 
                 custom_prob = [], putModels = 0 , X = []):
            
             """              
                 COMPUTES METRICS OF THE ALGORITHM
                 Acuraccy, Balanced acuraccy, Auc, Precision,
                 RSpecificity, Sensitivity,  TP, TN, FP, FN,
                 Percentage of High Cost Patients
                 Percentage of Low Cost Patients
                 
                  
                 y: training or testing labels 
                 tau: Threshold for probabilities
                 custom_prob: Probabilities produced by the model
                              based  on which you want to calculate
                              the class, these correspond
                              for a datapoint to belong to class 1
                 putModels:  Checks if you put  model to do the predictions
                             or the probabilities for each data point 
                             to belong to  class 1.
                     
             """
             if  putModels != 0 :
                 probabilities = model.predict_proba( X )[:,1]
            
             else:
                 
                 probabilities = custom_prob
                
                    
             auc = roc_auc_score( y , probabilities)  
             roc = roc_curve(y, probabilities)
             
             #Calculate tau if calc_tau is 1
             #Given we have provided probability matrix
            
             
             #THRESHOLDING BASED ON TAU IN ORDER TO GET THE 
             #ESTIMATED LABELS FOR EACH DATAPOINT
             probabilities[ np.where( probabilities >= tau ) ] = 1
             probabilities[ np.where( probabilities < tau ) ] = 0
             predictions = probabilities
              
             #METRICS CALCULATION
             precision =  precision_score(y, predictions) #CALCULATE THE PRECISION
             sensitivity = recall_score(y, predictions)  #CALCULATE THE RECALL
             accuracy = accuracy_score(y, predictions) #CALCULATE THE ACCURACY
             bal_acc = balanced_accuracy_score(y, predictions) #CALCULATE THE BALANCED ACCURACY
             f1 = f1_score(y, predictions)
             
             clusterSize = len( y )  #Cluster Size
             highCostPerc = len( np.where( y == 1)[0] )/clusterSize
             lowCostPerc = len( np.where( y == 0)[0] )/clusterSize
             
             
             TP = len( np.where(  (y == 1) * (predictions == 1) )[0] )
             TN = len( np.where(  (y == 0) * (predictions == 0) )[0] )
             
             FP = len( np.where(  (y == 0) * (predictions == 1) )[0] )
             
             FN = len( np.where(  (y == 1) * (predictions == 0) )[0] )
             
             #print(TP, TN, FP, FN, clusterSize)
             
             specificity = TN/(FP + TN)
             FPR = 1 - specificity
             
             #PUT ALL THE METRICS IN A LIST AND RETURN THEM
             metrics =  [cluster, clusterSize, highCostPerc, lowCostPerc,
                         TP, TN, FP, FN,
                         FPR, specificity, sensitivity, precision, 
                         accuracy, bal_acc, f1, auc]
             
             return metrics, roc
         
            
def logistic_cluster(Xtrain = [], Xtest = [], ytrain = [], ytest = [],
                                          n_clusters = 5,  
                                          n_jobs = -1, testlabels = [],
                                          labels = [], Cs = [10], 
                                          cv = 2, tau = 0.5, scoring = 'f1',
                                          penalty = 'l1',
                                          solver = 'saga',max_iter = 100):
            
            """ Performs logistic Regression in each cluster 
            
            Xtrain: TRAINING DATA: DEFAULT: xTrain
            n_clusters: #CLUSTERS THE DATA HAVE BEEN CLUSTERED TO
            n_jobs: #JOBS
            labels: LABELS DEPICTING EACH DATAPOINT"S CLUSTER 0-N_CLUSTERS
            Cs: INVERSEn REGULARIZATION PARAMETER , DEFAULT: 10
            
            
            """
           
            # matrix for logistic regression weights for each cluster
            weights = []   
            # matrix with metrics for each cluster
            metrics = []   
            #metrics for test  data in each cluster
            metricsTest = []  
            
            #If we give or not test data
            gate = len(testlabels)
            
            
            columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
            models = []
            
            #PERFORM LOGISTIC REGRESSION TO EACH CLUSTER
            for  cluster in np.arange( n_clusters ):  
                #INDEXES OF CLUSTER "cluster"
                inC = np.where( labels == cluster )[0] 
                #POSITIVE INDEXES OF CLUSTER "cluster"
                indpos = np.where( ytrain[inC] == 1 )[0]  
                #NEGATIVE INDEXES OF CLUSTER "cluster"
                
                indneg = np.where( ytrain[inC] == 0 )[0]  
                
                if gate > 0 : #handling test data
                    inCtest = np.where( testlabels == cluster )[0]
                  
                    
               #IF WE HAVE LESS THAN TWO POSITIVE OF NEGATIVE DATAPOINTS 
               #APPEND ZEROS
                if  indpos.size < 2 or indneg.size < 2 :  
                    
                    weights.append( [0, 0] )
                    metrics.append( [cluster, 1, 1, 1, 1, 1] )
                    metricsTest.append( [cluster,1, 1, 1, 1, 1] )
                    
                #ELSE PERFORM LOGISTIC REGRESSION TO EACH CLUSTER SEPARATELY  
                else:  
                    
                    #PERFORM L1 LOGISTIC REGRESSION
                    sgd = LogisticRegressionCV(Cs = Cs, penalty = penalty,
                             scoring = scoring, solver = solver,
                             max_iter = max_iter,
                             cv = cv).fit( Xtrain[inC], ytrain[inC])
                             
                    #CALCULATE METRICS : ACCURACY, RECALL, PRECISION , BALANCED ACCURACY
                   
                    metCluster , _= calc_metrics( model = sgd, X = Xtrain[inC],
                                 y = ytrain[inC], cluster = cluster, tau = tau,
                                 putModels = 1)
                    #APPEND THE CLUSTERS WEIGHTS EXTRACTED FROM THE CLASSIFIER
                   
                   
                    #APPEND METRICS
                    
                    if len( inC ):
                        
                        metrics.append( metCluster )
                        
                    else:
                        
                        metrics.append( [cluster,-1, -1, -1, -1] )
                        
                    #SAVE SGD MODELS FOR TEST DATA
                    models.append( sgd )
                    #evaluate test data
                    #if we have provided  test labels and data
                    if gate > 0 :  
                        if len( inCtest ):
                         
                            metTest, _ = calc_metrics( model = sgd, 
                                                      X = Xtest[inCtest],
                                 y = ytest[inCtest], cluster = cluster, 
                                 tau = tau,
                                 putModels = 1)
                            
                            metricsTest.append( metTest )
                            
                        else:
                            metricsTest.append([cluster, -1, -1, -1, -1])
                            #print("I am here Test")
                        
                    
                    
                        
            metrics = pd.DataFrame ( metrics, columns = columns ) #MAKE METRICS INTO A DATAFRAME
            
            if gate: #check if test data are calculated
                #print('HERE')
                metricsTest = pd.DataFrame( metricsTest, columns = columns ) #MAKE METRICS INTO A DATAFRAME
                
            return metrics,   metricsTest, weights, models