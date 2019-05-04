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
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, \
 balanced_accuracy_score, f1_score
#from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve






def CalculateSoftLogReg( models = [],  Xtrain = [], Xtest = [], ytrain = [],
                        ytest = [],  membTrain = [], membTest = []):
                         
            """
                This Function probably is not gonna be used in the
                future. I might modify it or leave it at some point
                The initial purpose was to calculate metrics for the
                Supervised Gaussian Mixtures Models. But I have incorporate
                the functionality inside the class
                
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
        
        
def optimalTau(probabilities, ylabels, returnAll = 0, mode = 0, 
               targetMax = 0, targetValue = 0.2):
            
            """ Finds the Optimal tau based on the F1 score or precision or
                recall
                Input: Probabilities of train data of being class 1,
                ylabels of training data
                
                returnAll: 0 or 1 :Default 0, If 1 return a more complete
                set of parameters except tau
                mode: 0 (maximize F1 score), 1 (maximize precision), 2
                      (maximize) recall
                targetMax: We included one more parameter if targetMax is 0
                then our Target is precision  if our targetMax is 1 then the 
                target is recall
                targetValue:  maximize targetMax (precision or Recall) such that
                the target value (recall , precision) is at least "targetValue"
                """
            
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
            prec = precision
            rec = recall
            #list with f1, precision , recall
            metOld = [f1, precision, recall]
            #precision recall list of lists
            precRec = [precision, recall]
            #tau that maximizes target with given targetvalue
            tauTarget = -1
            
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
                    metNew = [f1new, precision, recall]
                
                prob_F1.append( [probability, metNew[mode]] )   #thresholds with F1 scores if you want to draw a graph
                
                if targetMax == 0: #maximize precision with given recall
                    
                    if recall >= targetValue:
                        tauTarget = probability
                        precRec.append([precision, recall])
                
                else:
                    
                    if precision >=  targetValue:
                        tauTarget = probability
                        precRec.append([precision, recall])
                    
                
                #maximize  f1 precision or recall
                if metNew[mode] >= metOld[mode] :
                    threshold = probability
                    f1 = f1new
                    prec = precision  #you can return precision
                    rec = recall      #you can return  recall
                    metOld  = [f1, prec, rec]
                                        # these are mostly for correctness
                                        #checking
            
            #OUTSIDE THE LOOP
            if returnAll == 1:
                params = {'tau': threshold, 'curve': np.array(prob_F1), 
                          'precision': prec, 'recall': rec, \
                          'precRec': precRec, 'tauTarget': tauTarget}
                return params
            
            return threshold #, f1, np.array(prob_F1), prec, rec
        
def calc_metrics(model = None, cluster = -1, y = None, tau = 0.5, 
                 custom_prob = None, putModels = 0 , X = None):
            
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
def predict_y(probabilities, tau = 0.5):
     """return the predictions given probabilities and thresholds """
    
     probabilities[ np.where( probabilities >= tau ) ] = 1
     probabilities[ np.where( probabilities < tau ) ] = 0
     predictions = probabilities
     
     return predictions
    
         
            
def metrics_cluster(models = None, ytrain = None, ytest = None,
                                          testlabels = None,
                                          trainlabels = None,
                                          Xtrain = None,  Xtest = None):
            """
             Calculates Metrics such as accuracy, balanced accuracy,
             specificity, sensitivity, precision, True Positives,
             True Negatives etc.
             
             These metrics are calculated for each cluster:
             models: predictive models trained in each cluster
             ytrain: Target labels of training set 
             ytest: target labels of test set
             testlabels: a matrix with numbers from 0 to c-1 number of clusters
                         indicating in which cluster each data point belongs
                         in the test set
             trainlabels: the same as testlabels but for training data
             Xtrain: trainiing data
             Xtest: testing data
                     
            
            
            """
           
           
            # matrix with metrics for each cluster
            metricsTrain = []   
            #metrics for test  data in each cluster
            metricsTest = []  
            
            
            
            
            columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
            
            
            #Calculate the Metrics for Each Cluster
            for  cluster in np.arange( len( models ) ):  
                #INDEXES OF CLUSTER "cluster"
                inC = np.where( trainlabels == cluster )[0] 
                inCT = np.where( testlabels == cluster )[0]
                
                #predict probabilities  of data in cluster "cluster"
                #to be 1
                probTrain = models[cluster].predict_proba(Xtrain[inC])[:, 1]
                probTest = models[cluster].predict_proba(Xtest[inCT])[:, 1]
                
                #calculate optimal tau based on F1
                tau = optimalTau(probTrain, ytrain[inC])
                    
                #CALCULATE METRICS : ACCURACY, RECALL, PRECISION ,
                #BALANCED ACCURACY ETC
                metTrain , _= calc_metrics(  custom_prob = probTrain, 
                                                  y = ytrain[inC], 
                                                  cluster = cluster, 
                                                  tau = tau )
                
                metTest, _ = calc_metrics(  custom_prob = probTest,
                                                  y = ytest[inCT], 
                                                  cluster = cluster, 
                                                  tau = tau)
                
                metricsTrain.append( metTrain )
                metricsTest.append( metTest )
                    
                   
                   
                   
            
                    
                        
                    
                    
            #Create a dataframe with metrics for better Visualization         
            metricsTrain = pd.DataFrame ( metricsTrain, columns = columns )
            metricsTest = pd.DataFrame( metricsTest, columns = columns ) 
                
            return metricsTrain, metricsTest