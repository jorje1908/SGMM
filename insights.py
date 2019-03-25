#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:57:54 2019

@author: george
"""

import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, \
 balanced_accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve


#from scipy.stats import multivariate_normal


class Insights():
        """ This class Takes a data Set and defines multiple manipulations 
        on it """
        
        
        
        def __init__(self, data):
            """ Initializing some basic Attributes of the class 
            Xdim : X dimensions of data Matrix
            Ydim : Y dimensions of data Matrix
            columns : column names of data matrix
            data : numpy array of data
            target : numpy array of target values
            Xtrain, Xtest, Ytrain, Ytest : training, testing matrices
            Trdims = dimensions X, Y of train matrix
            Testdims : dimensions X, Y of test data
            ind2col : a dictionary with index to column name correspondance
            """
            
            self.Xdim = data.shape[0]
            self.Ydim = data.shape[1] - 1
            self.columns = data.columns.tolist()
            self.data = data.iloc[:,:-1].values   #last column is target 
            self.target = data['Target'].values
            mats = self.train_test_split0()
            self.data = []
            self.Xtrain = mats[0]
            self.Xtest = mats[1]
            self.ytrain = mats[2]
            self.ytest = mats[3]
            self.Trdims = self.Xtrain.shape
            self.Testdims = self.Xtest.shape
            self.ind2col = self.ind2Col()
            
            
            
            
        
        def ind2Col(self):  #matches numerical column index  with column names
            x = [b for b in range(self.Ydim)]
            diction = dict(zip(x, self.columns))
            
            return diction
        
        
        
        
    
        #preprocessing
        def train_test_split0(self):
            """ splits data to training and testing data"""
            
            Xtrain, Xtest, ytrain, ytest = train_test_split(self.data,
                                                  self.target, test_size = 0.2,
                                                  random_state = 1512)
            
            return Xtrain, Xtest, ytrain, ytest
        
        
        
        
        #ALGORITHMS
        #Kmeans algorithm 
        def k_means(self, Xtrain = [], Xtest = [], n_jobs = -1,
                    n_clusters = 5, both = 1):
            
            """ 
            Implements KMeans algorithm
            n_clusters: #clusters default:5
            n_jobs: #jobs default:2
            Xtrain: Training data, default: Xtrain of the class
            both: fit both train and test data an a whole dataset
            
            """
            if not len(Xtrain): 
                 Xtrain  = self.Xtrain
                
            if not len(Xtest):
                 Xtest = self.Xtest
                
            if (both == 0):  #apply K-means Clustering in both training and 
                            #testing data 
                
                kmeans = KMeans(n_clusters = n_clusters, random_state = 0,
                                n_jobs = n_jobs).fit(Xtrain) #scikit learn 
                                                             #kmeans algorithm fits only train
            else:
                
                X = np.concatenate((Xtrain, Xtest), axis = 0)
                kmeans = KMeans(n_clusters = n_clusters, random_state = 0,
                                n_jobs = n_jobs).fit(X) #scikit learn kmeans algorithm fit both train test
        
            return kmeans #returns the object of kmeans
        
        
        
        
        
        #L1 logistic regression algorithm with 
        #stochastic gradient descebt optimizer
        def logistic_regr( self, Xtrain = [], ytrain = [], n_jobs = -1,
                          alpha = 0.001, sample_weight = None, 
                          cv = None, calc_tau = None):
            """ logistic regression classifier with stochastic gradient descent
            Xtrain: training data, default classes data
            Ytrain: training target, default classes data
            n_jobs: #jobs, default:2
            alpha: REGULARIZATION PARAMETER, DEFAULT : 0.0001
            sample weight = weights for the training
            cv:  cross validattions fold
            """
            
            ### ADDED CROSS VALIDATION
            if not len( Xtrain ):
                Xtrain = self.Xtrain
                
            if not len( ytrain ):
                ytrain = self.ytrain
            
            if cv is not None :
            
                sgd = SGDClassifier( loss = "log", penalty = 'l1',
                                    n_jobs = n_jobs, early_stopping = True, 
                                    alpha = alpha, random_state = 0 )
                
                param_grid = {'alpha' : [0.1,  0.01,
                                         0.001, 0.0001]}
            
                clf = GridSearchCV( sgd, param_grid = param_grid, 
                                   n_jobs = n_jobs, 
                               scoring = 'f1', cv = cv).fit(Xtrain, 
                                ytrain,
                                sample_weight = sample_weight )
            
               
            
            else :
                 clf = SGDClassifier( loss = "log", penalty = 'l1', 
                                     n_jobs = n_jobs, 
                                     early_stopping = True, alpha = alpha, 
                                     random_state = 0 ).fit(Xtrain,
                                     ytrain, sample_weight = sample_weight)
                
            
            
            probabilities = clf.predict_proba( Xtrain )[:,1]
            
            if calc_tau is not None:
                
                tau = self.optimalTau( probabilities, ytrain )
                
            else:
                
                tau = None
            
            
            return clf, tau #returns the classifier and the Optimal Threshold based on the F1 score
        
        
        
        def logistic_regrReal( self, Xtrain = [], ytrain = [], n_jobs = -1,
                          alpha = 0.001, sample_weight = None, 
                          cv = None, calc_tau = None):
            """ logistic regression classifier with stochastic gradient descent
            Xtrain: training data, default classes data
            Ytrain: training target, default classes data
            n_jobs: #jobs, default:2
            alpha: REGULARIZATION PARAMETER, DEFAULT : 0.0001
            sample weight = weights for the training
            cv:  cross validattions fold
            """
            
            ### ADDED CROSS VALIDATION
            if not len( Xtrain ):
                Xtrain = self.Xtrain
                
            if not len( ytrain ):
                ytrain = self.ytrain
            
           
            
            clf = LogisticRegressionCV(Cs = [0.0001, 0.001, 0.01], penalty = 'l1',
                                            n_jobs = n_jobs, 
                                            random_state = 0,
                                            scoring = 'f1', cv = 2 , solver = 'saga', 
                                            refit = True).fit(Xtrain, ytrain, sample_weight)
                
                
            
                
                
            
            
            probabilities = clf.predict_proba( Xtrain )[:,1]
            
            if calc_tau is not None:
                
                tau = self.optimalTau( probabilities, ytrain )
                
            else:
                
                tau = None
            
            
            return clf, tau #returns the classifier and the Optimal Threshold based on the F1 score
        
        
        
        #KMEANS ALONG WITH L1 LOGISTIC REGRESSION WITHIN THE CLUSTER
        def Kmeans_LogRegr( self, Xtrain = [], Xtest = [], ytrain = [], ytest = [],  
                           n_jobs = -1, n_clusters = 5, alpha = 0.0001,
                           both = 1, cv = None, calc_tau = None, name = None ):
            """ Performs Logistic Regression  in the clusters produced from Kmeans 
            Xtrain: TRAINING DATA, DEFAULT CLASSE'S Xtrain
            ytrain: TRAINING TARGET DATA
            n_jobs: #JOBS, DEFAULT:2
            n_clusters: #CLUSTERS, DEFAULT:5
            alpha: REGULARIZATION PARAMETER, DEFAULT: 0.0001
            
            
            
            """
            
            
            if not len( Xtrain ): #if no dataset specified  use the default
                Xtrain = self.Xtrain
                
            if not len( Xtest ):
                Xtest = self.Xtest
                
            if not len( ytrain ):
                ytrain = self.ytrain
            
            if not len(ytest):
                ytest = self.ytest
            
                
            kmeans  = self.k_means( Xtrain = Xtrain, 
                                   Xtest = Xtest, n_jobs = n_jobs, 
                                   n_clusters = n_clusters, both = both )  #RUN  K MEANS FIRST
            
            labels = kmeans.labels_    #extract cluster labels
            
            metrTr, weights, metrTest, model = self.logistic_cluster( Xtrain = Xtrain, 
                                                Xtest = Xtest, ytrain = ytrain,
                                                ytest = ytest, n_clusters = n_clusters,
                                                n_jobs = n_jobs, labels = labels[0:Xtrain.shape[0]],
                                                     testlabels = labels[Xtrain.shape[0]:], alpha = alpha, cv = cv, calc_tau = calc_tau )
            
            hCostStatsTr, lCostStatsTr = self.getClusterStats(y = ytrain, labels = labels[ 0 : ytrain.shape[0] ], n_clusters = n_clusters)
            hCostStatsTest, lCostStatsTest = self.getClusterStats(y = ytest, labels = labels[ ytrain.shape[0]: ], n_clusters = n_clusters)
            
            
           # print(metrTr.shape, metrTest.shape, hCostStatsTest.shape)
            metrTr = self.addTotalBalanced( metrTr,  hCostStatsTr['Cluster_pop_%'].values)  #adding the total balanced accuracy based on the statistics of each cluster
            metrTest = self.addTotalBalanced( metrTest,  hCostStatsTest['Cluster_pop_%'].values )
            
            methodName = name
            topNumber = 700
            
            nameList, tfIdf = self.CreateCloudsClusters(np.concatenate((Xtrain, Xtest), axis = 0), self.columns, topNumber, 
                                                        methodName, labels, n_clusters)
            
            text1, text2 = self.CreateCloudsClustersWeights(self.columns, topNumber, methodName, n_clusters, weights)
            
            wordCloudIns = {'labels': labels, 'columns': self.columns, 'topNumber': topNumber, 'clusters': n_clusters, 
                            'weights': weights}
            
            return metrTr, metrTest, hCostStatsTr, \
        lCostStatsTr, hCostStatsTest, lCostStatsTest,\
        weights, model, nameList, tfIdf, wordCloudIns
        
        
        
        
        
        
        #PERFORMS LOGISTIC L1 LOGISTIC REGRESSION TO EACH CLUSTER OF CLUSTERED DATA 
        #SEPARATELY
        
        def logistic_cluster(self, Xtrain = [], Xtest = [], ytrain = [], ytest = [], n_clusters = 5,  
                                          n_jobs = -1, testlabels = [], labels = [], alpha = 0.0001, 
                                          cv = None, calc_tau = None):
            
            """ Performs logistic Regression in each cluster 
            
            Xtrain: TRAINING DATA: DEFAULT: xTrain
            n_clusters: #CLUSTERS THE DATA HAVE BEEN CLUSTERED TO
            n_jobs: #JOBS
            labels: LABELS DEPICTING EACH DATAPOINT"S CLUSTER 0-N_CLUSTERS
            alpha: REGULARIZATION PARAMETER , DEFAULT: 0.0001
            
            
            """
            if not len( Xtrain ): #if no dataset specified  use the default
                Xtrain = self.Xtrain
                
            if not len( Xtest ):
                Xtest = self.Xtest
                
            if not len( ytrain) :
                ytrain = self.ytrain
                
            if not len( ytest ):
                ytest = self.ytest
            
            weights = []   # matrix for logistic regression weights for each cluster
            metrics = []   # matrix with metrics for each cluster
            metricsTest = []  #metrics for test  data in each cluster
            gate = len(testlabels)
            
            
            columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
            models = []
            
            for  cluster in np.arange( n_clusters ):  #PERFORM LOGISTIC REGRESSION TO EACH CLUSTER
                
                inC = np.where( labels == cluster )[0]   #INDEXES OF CLUSTER "cluster"
                indpos = np.where( ytrain[inC] == 1 )[0]  #POSITIVE INDEXES OF CLUSTER "cluster"
                indneg = np.where( ytrain[inC] == 0 )[0]  #NEGATIVE INDEXES OF CLUSTER "cluster"
                
                if gate > 0 : #handling test data
                    inCtest = np.where( testlabels == cluster )[0]
                   # print("Gate is Good")
                    #print(len(inCtest))
                    
              
                if  indpos.size < 2 or indneg.size < 2 :  #IF WE HAVE LESS THAN TWO POSITIVE OF NEGATIVE DATAPOINTS APPEND ZEROS
                    
                    weights.append( [0, 0] )
                    metrics.append( [cluster, 1, 1, 1, 1, 1] )
                    metricsTest.append( [cluster,1, 1, 1, 1, 1] )
                   
                else:  #ELSE PERFORM LOGISTIC REGRESSION TO EACH CLUSTER SEPARATELY
                    
                    #PERFORM L1 LOGISTIC REGRESSION
                    sgd, tau = self.logistic_regr( Xtrain[inC], ytrain[inC], n_jobs = n_jobs, alpha = alpha, cv = cv, calc_tau = calc_tau )
                    #CALCULATE METRICS : ACCURACY, RECALL, PRECISION , BALANCED ACCURACY
                   
                    metCluster , _= self.calc_metrics( sgd, Xtrain[inC], ytrain[inC], cluster = cluster, tau = tau )
                    #APPEND THE CLUSTERS WEIGHTS EXTRACTED FROM THE CLASSIFIER
                   
                    if cv is None:
                        
                        weights.append( sgd.coef_ )
                    else:
                        
                        weights.append( sgd.best_estimator_.coef_ )
                    #APPEND METRICS
                    
                    if len( inC ):
                        
                        metrics.append( metCluster )
                        
                    else:
                        
                        metrics.append( [cluster,-1, -1, -1, -1] )
                        
                    #SAVE SGD MODELS FOR TEST DATA
                    models.append( sgd )
                    #evaluate test data
                    if gate > 0 :  #if we have provided 
                        if len( inCtest ):
                          #  print("I am here \n")
                            metTest, _ = self.calc_metrics( sgd, Xtest[inCtest], ytest[inCtest], cluster = cluster, tau  = tau)
                            metricsTest.append( metTest )
                            
                        else:
                            metricsTest.append([cluster, -1, -1, -1, -1])
                            #print("I am here Test")
                        
                    
                    
                        
            metrics = pd.DataFrame ( metrics, columns = columns ) #MAKE METRICS INTO A DATAFRAME
            
            if gate: #check if test data are calculated
                
                metricsTest = pd.DataFrame( metricsTest, columns = columns ) #MAKE METRICS INTO A DATAFRAME
                
            return metrics,  weights, metricsTest, models
        
        
        
        
        
        #SUPERVISED CLUSTERING ALGORITHM
        
        def supervised_clustering(self, Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], memb_init = [],  n_clusters = 5, n_jobs = -1, 
                                  alpha = 0.01, max_iter = 20,  tol = 10**(-3), version = 1,
                                  cv = None, calc_tau = None):
    
           """ SUPERVISED CLUSTERING ALGORITHM
               LOGISTIC REGRESSION AS THE SUPERVISED DRIVER
               
               Xtrain: TRAINING DATA, DEFAULT Xtrain
               ytrain: TARGET TRAINING DATA, DEFAULT Ytrain
               n_clusters: #CLUSTERS TO ORGANIZE THE DATA
               n_jobs: #JOBS
               alpha: REGULARIZATION PARAMETER , DEFAULT: 0.0001
               max_iter: MAXIMUM NUMBER OF ITERATIONS TO CONVERGANCE
               tol: TOLERANCE OF THE ALGORITHM
               
               THIS FUNCTION REQUIRES THE ADDITIONAL FUNCTIONS: initializeMember AND calculateStats, SCIKIT LEARN SGD, KMEANS
               """
    
           if not len(Xtrain): #if no dataset specified  use the default
                Xtrain = self.Xtrain
                
           if not len(Xtest):
                Xtest = self.Xtest
                
           if not len(ytrain):
                ytrain = self.ytrain
           
           if not len(ytest):
                ytest = self.ytest
           
            
           if not len(memb_init): #KMEANS INITIALIZATION
               
               #FIRST PERFORM K MEANS TO INITIALIZE THE MEMBERSHIP TABLE
               kmeans = KMeans(n_clusters = n_clusters, n_jobs = n_jobs, random_state = 0).fit(Xtrain)
               labels = kmeans.labels_    #TAKE THE LABELS OF THE KMEANS
    
               memberships = self.initializeMember( labels, n_clusters )    #CREATE INITIAL MEMBERSHIPS BASED ON THE KMEANS CLUSTERING
               
           else: #CUSTOM INITIALIZATION
               memberships = memb_init
           
           ytrain.astype( int )
           indexing = np.arange( Xtrain.shape[0] )
           errors = np.zeros( [ Xtrain.shape[0], n_clusters ] ) #ERROR MATRIX TO UPDATE MEMBERSHIPS
           
    
           for iteration in np.arange( max_iter ):  #outside optimization loop
              # prob = []
               models = []
               for i in np.arange( n_clusters ):  #FOR EACH CLUSTER
                   sgd, _ = self.logistic_regr(Xtrain = Xtrain, ytrain = ytrain, n_jobs = n_jobs, alpha = alpha, 
                                                        sample_weight = memberships[:,i] + 10**(-6),
                                                        cv = cv, calc_tau = calc_tau)    #FIT AN L1 CLASSIFIER FOR EACH CLUSTERS WITH 
                                                                                                        #RESPECTIVE MEMBERSHIP WEIGHTS
                   models.append( sgd )
           
                   predict_prob = sgd.predict_proba( Xtrain )   #PREDICT THE PROBABILITY EACH DATA POINT TO BELONG TO EACH CLASS 0 OR 1
                  # prob.append(predict_prob) NOT NEEDED IN THE FINAL ALGORITHM
                  # for j in np.arange(Xtrain.shape[0]): #FOR EACH DATA POINT UPDATE THE MEMBERSHIP TO BELONG IN CLUSTER i ACCORDING TO EACH LABEL
                   #    errors[j,i] = predict_prob[j, int(ytrain[j])]
                   errors[:,i] = predict_prob[ indexing , ytrain]
                
    
               ### all memberships have been established - all errors
               #update memeberships
               newmemb = ( ( errors.T + 10**(-6)/errors.shape[1] )/ (np.sum(errors, axis = 1) + 10**(-6) ) ).T
               diff = np.sum(np.abs(newmemb - memberships))/newmemb.shape[0]
               memberships = newmemb
        
               print( "iteration: {}, diff: {}".format(iteration, diff) )
               if diff < tol:
                  break
        
           lab = np.argmax(memberships, axis = 1) #FIND THE FINAL CLUSTERING LABELS
           
           if version == 1: #version control
               
               w, m, cov = self.calculateStats(Xtrain, memberships) #CALCULATE PROB, MEANS COVARIANCES
               
           else:
               w = []
               m = []
               cov = []
           
           return  lab, memberships, models,  w, m, cov
       
        
        
        
        #SUPERVISED CLUSTERING ALGORITHM WITH L1 LOGISTIC REGRESSION
        #SIMILAR TO UNSUPERVISED CLUSTERING WITH L1 LOGISTIC REGRESSION
        def SupClust_LogRegr(self, Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = 2, 
                             n_clusters = 5, alpha = 0.0001, max_iter = 20, tol = 10**(-3) ):
            
            """ SUPERVISED CLUSTERING ALGORITHM 
            
                <<<THIS WILL NOT USED SOMEWHERE IN THIS PROJECT >>>>>
                
            Xtrain: TRAINING DATA, DEFAULT: Xtrain
            ytrain: TRAINING TARGET DATA, DEFAULT: ytrain
            n_jobs: #JOBS
            n_clusters: #CLUSTERS , DEFAULT:5
            alpha: REGULARIZATION PARAMETER, DEFAULT: 0.0001
            max_iter: MAXIMUM #ITERATIONS OF SUPERVISED CLUSTERING ALGORITHM, DEFAULT 20
            tol: TOLERANCE OF CONVERGANCE OF SUPERVISED CLUSTERING ALGORITHM, DEFAULT: 10^(-3)
            
            """
            
            
            if not len(Xtrain):
                Xtrain = self.Xtrain
                
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
            
            if not len(ytest):
                ytest = self.ytest
            
                
            labels, mem ,_,_,_ = self.supervised_clustering(Xtrain = Xtrain, ytrain = ytrain, n_jobs = n_jobs, n_clusters = n_clusters,
                                                 alpha = alpha, max_iter = max_iter, tol = tol) 
                                                ### RUN SUPERVISED CLUSTERING ALGORITHM FIRST TO GET THE LABELS
           
            high, low = self.getClusterStats(ytrain, labels, n_clusters)   #GET THE STATISTICS OF EACH CLUSTER
            
            metrics, weights, _, _ = self.logistic_cluster(Xtrain = Xtrain, ytrain = ytrain, n_clusters = n_clusters,
                                                     n_jobs = n_jobs, labels = labels, alpha = alpha) #PERFORM LOGISTIC REGRESSION TO EACH CLUSTER SEPAATELY
            
            return metrics, weights, high , low  #return metrics of L1 LR, weights, highcost Statistics, low cost Statistics
        
        
        
        
        
        #Combination of Supervised and Unsupervised Clustering
        #Logistic Regression L1 in both test and training data
        #Results Based on the results Analytics
        def SupClust_LogRegrGMM(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = 2,
                                n_clusters = 5, alpha = 0.0001, max_iter = 10, tol = 10**(-3), max_iter2 = 10, tol2 = 10**(-3) ):
            
            
            """ Supervised and Unsupervised Clustering
                WE WILL USE THE VERSION 2 of this
                
                Xtrain: TRAINING DATA, Default Xtrain
                ytrain: TRAINING TARGET DATA, DEFAULT ytrain
                n_jobs: #JOBS, DEFAULT 2
                n_clusters: #CLUSTERS, DEFAULT:5
                alpha: REGULARIZATION PARAMETER: DEFAULT:0.0001
                max_iter: MAXIMUM NUMBER OF ITERATIONS OF THE SUPERVISED CLUSTERING ALGROITHM
                tol: TOLERANCE OF THE SUPERVISED ALGORITHM """
                
            #TAKE DEFAULT IF NO INPUT   
            if not len(Xtrain):
                Xtrain = self.Xtrain
            
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
                
            if not len(ytest):
                ytest = self.ytest
            
            #We return the labels of train and test data So we can perform logistic regression in the  clusters
            #and evaluate the performance of each cluster in the testing data
            gmm, Trlabels, Testlabels, membTr, membTest = self.Gmm_Supervised(Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain, ytest = ytest, n_jobs = n_jobs,
                                                            n_clusters = n_clusters,
                                                            alpha = alpha, max_iter = max_iter, tol = tol, 
                                                            mix = 0.5, max_iter2 = max_iter2,
                                                            tol2 = tol2)
            
            #TO BE CONTINUED
            highTr, lowTr = self.getClusterStats(ytrain, Trlabels, n_clusters)
            highTest, lowTest = self.getClusterStats(ytest, Testlabels, n_clusters)
            
            metrics, weights, metricsTest, models = self.logistic_cluster(Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain, ytest = ytest,  n_clusters = n_clusters,
                                                     n_jobs = n_jobs, testlabels = Testlabels, labels = Trlabels, alpha = alpha) #PERFORM LOGISTIC REGRESSION TO EACH CLUSTER SEPAATELY
            
            
            softMetricsTr, softMetricsTest = self.CalculateSoftLogReg( Xtrain = Xtrain, 
                                                                      Xtest = Xtest, membTrain = membTr,
                                                                      ytrain = ytrain, ytest = ytest,
                                                                      membTest = membTest )
            
            metrics = self.addTotalBalanced( metrics,  highTr['Cluster_pop_%'].values)  #adding the total balanced accuracy based on the statistics of each cluster
            metricsTest = self.addTotalBalanced( metricsTest,  highTest['Cluster_pop_%'].values )
            
            return softMetricsTr, softMetricsTest,  metrics, metricsTest, highTr, highTest, weights,membTr, membTest, models
        
        
        
            
            
        def SupClust_LogRegrGMM_V2(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = -1,
                                n_clusters = 5, alpha = 0.0001, max_iter = 10, tol = 10**(-3), 
                                max_iter2 = 10, tol2 = 10**(-3), lambd = 0.5,
                                cv = None, calc_tau = None, name = None):
            
            
            """ Supervised and Unsupervised Clustering
                IT CALLS Gmm_Supervised_V2 --> the supervised unsupervised algorithm
                         logistic_Cluster  --> algorithm to perform and evaluate clusters with logistic regression
                
                Xtrain: TRAINING DATA, Default Xtrain
                ytrain: TRAINING TARGET DATA, DEFAULT ytrain
                n_jobs: #JOBS, DEFAULT 2
                n_clusters: #CLUSTERS, DEFAULT:5
                alpha: REGULARIZATION PARAMETER: DEFAULT:0.0001
                max_iter: MAXIMUM NUMBER OF ITERATIONS OF THE SUPERVISED CLUSTERING ALGROITHM
                tol: TOLERANCE OF THE SUPERVISED ALGORITHM """
                
            #TAKE DEFAULT IF NO INPUT   
            if not len(Xtrain):
                Xtrain = self.Xtrain
            
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
                
            if not len(ytest):
                ytest = self.ytest
            
            #We return the labels of train and test data So we can perform logistic regression in the  clusters
            #and evaluate the performance of each cluster in the testing data
            softModels, Trlabels, Testlabels, membTr, membTest = self.Gmm_Supervised_V2( Xtrain = Xtrain, Xtest = Xtest, 
                                                                                      ytrain = ytrain, ytest = ytest, n_jobs = n_jobs,
                                                                                      n_clusters = n_clusters, alpha = alpha, max_iter = max_iter, 
                                                                                      tol = tol, mix = 0.5, max_iter2 = max_iter2,
                                                                                      tol2 = tol2, lambd = 0.5, cv = cv, calc_tau = calc_tau)
            
            #TO BE CONTINUED
            highTr, lowTr = self.getClusterStats(ytrain, Trlabels, n_clusters)
            highTest, lowTest = self.getClusterStats(ytest, Testlabels, n_clusters)
            
            metrics, weights, metricsTest, models = self.logistic_cluster(Xtrain = Xtrain, Xtest = Xtest, 
                                                                          ytrain = ytrain, ytest = ytest,
                                                                          n_clusters = n_clusters,
                                                                          n_jobs = n_jobs, testlabels = Testlabels, 
                                                                          labels = Trlabels, alpha = alpha, 
                                                                          cv = cv, calc_tau = calc_tau)     #PERFORM LOGISTIC 
                                                                                                            #REGRESSION TO EACH CLUSTER SEPARATELY
                                                                                                            
            softMetricsTr, softMetricsTest, roc1, roc2 = self.CalculateSoftLogReg(models = models, Xtrain = Xtrain, 
                                                                      Xtest = Xtest, membTrain = membTr,
                                                                      ytrain = ytrain, ytest = ytest,
                                                                      membTest = membTest, tau = 0)
            
            metrics = self.addTotalBalanced( metrics,  highTr['Cluster_pop_%'].values)  #adding the total balanced 
                                                                                        #accuracy based on the statistics of each cluster
            metricsTest = self.addTotalBalanced( metricsTest,  highTest['Cluster_pop_%'].values )
            
            labels = np.concatenate((Trlabels, Testlabels), axis = 0)
            
            
            methodName = name
            topNumber = 700
            nameList, tfIdf = self.CreateCloudsClusters(np.concatenate((Xtrain, Xtest), axis = 0), 
                                                        self.columns, topNumber, methodName, labels, n_clusters)
            
            textPos, textNeg = self.CreateCloudsClustersWeights(self.columns, topNumber, methodName, n_clusters, weights)
            
            clouds = {'labels': labels, 'columns': self.columns, 'topNumber': topNumber, 'clusters': n_clusters, 
                            'weights': weights, 'roc1': roc1, 'roc2': roc2}
            
            
            
            
            return  softMetricsTr, softMetricsTest, metrics, metricsTest, highTr, highTest, weights, membTr, membTest, models, clouds, tfIdf
        
        
        
        
        
        def CalculateSoftLogReg(self, models = [],  Xtrain = [], Xtest = [], ytrain = [], ytest = [],  membTrain = [],
                                membTest = [], cv = None, calc_tau = None, tau = None):
            """
                Calculates The soft clustering metrics """
                
            clusters = membTrain.shape[ 1 ]
                
           
            probTest = np.zeros( [Xtest.shape[0] ] )
            probTrain = np.zeros( [Xtrain.shape[0] ] )
            
            for i in np.arange( clusters ):
                    
                if not len(models): #for GMM
                    
                    model, tau = self.logistic_regr(Xtrain = Xtrain, ytrain = ytrain, cv = cv, calc_tau = calc_tau,
                                                  sample_weight = membTrain[:, i] )
                    
                else: #for Supervised Gmm
                    
                    model = models[i]
                
                probTest = probTest + model.predict_proba( Xtest )[ :, 1] * membTest[ :, i ]
                probTrain = probTrain + model.predict_proba( Xtrain )[ :, 1] * membTrain[ :, i ]
                
          
            columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
            if tau is not None:
                tau = self.optimalTau( probTrain, ytrain)  #calculate optimal tau on the training data
                print(' I am here TAU, {}'.format(tau))
            
            metricsTrain, roc1 = self.calc_metrics(y = ytrain, tau = tau, custom_prob = probTrain)
            metricsTest, roc2 = self.calc_metrics( y = ytest, tau = tau, custom_prob = probTest)
            
            metricsTrain = pd.DataFrame( [metricsTrain], columns = columns)
            metricsTest = pd.DataFrame( [metricsTest], columns = columns)
            
            return metricsTrain, metricsTest, roc1, roc2
        
        
        
        #PERFORM ROUNDS OF SUPERVISED ALONG WITH UNSUPERVISED  CLUSTERING
        def Gmm_Supervised(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = -1,
                           n_clusters = 5, alpha = 0.0001, max_iter = 10, tol = 10**(-3), mix = 0.5 , max_iter2 = 10, tol2 = 10**(-3), 
                           cv = None, calc_tau = None):
            
            """ SUPERVISED PLUS UNSUPERVISED CLUSTERING
                Xtrain: TRAINING DATA, Default Xtrain
                ytrain: TRAINING TARGET DATA, DEFAULT ytrain
                n_jobs: #JOBS, DEFAULT 2
                n_clusters: #CLUSTERS, DEFAULT:5
                alpha: REGULARIZATION PARAMETER: DEFAULT:0.0001
                max_iter: MAXIMUM NUMBER OF ITERATIONS OF THE SUPERVISED CLUSTERING ALGROITHM
                tol: TOLERANCE OF THE SUPERVISED ALGORITHM
                mix: MIXING COEFICIENT OF THE SUPERVISED AND UNSUPERVISED 
                max_iter2: Maximum times of supervised unsupervised clustering
                tol2: TOLERANCE FOR THE SUPERVISED UNSUPERVISED ALGORITHM
                """
            
            #TAKE DEFAULT IF NO INPUT   
            if not len(Xtrain):
                Xtrain = self.Xtrain
            
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
                
            if not len(ytest):
                ytest = self.ytest
                
            memberships = np.zeros([Xtrain.shape[0], n_clusters])
                
            for i in np.arange(max_iter2): #TIMES TO PERFORM THE ALGORITHM
                
                #RUN SUPERVISED CLUSTERING
                if i == 0:
                    labelsSC, membSC, models, weights, means, covariances_Inv = self.supervised_clustering(Xtrain = Xtrain,
                                                                                                   ytrain = ytrain,
                                                                                                   n_clusters = n_clusters,
                                                                                               n_jobs = n_jobs, max_iter = max_iter, tol = tol)
                else:
                    labelsSC, membSC, models,  weights,   means, covariances_Inv = self.supervised_clustering(Xtrain = Xtrain,
                                                                                                   ytrain = ytrain, 
                                                                                                   memb_init = memberships, n_clusters = n_clusters,
                                                                                               n_jobs = n_jobs, max_iter = max_iter, tol = tol)
                    
                #RUN UNSUPERVISED GMMS WITH INITIAL PARAMETERS GIVEN SUPERVISED TRAINING
                gmm = GaussianMixture(n_components = n_clusters, weights_init = weights,
                                      means_init = means, precisions_init =  covariances_Inv).fit(Xtrain)
                membershipsnew = gmm.predict_proba(Xtrain)
                
                error = np.sum( np.abs(memberships - membershipsnew) )/Xtrain.shape[0]
                memberships = membershipsnew
                
                print("GMM iteration: {}, error: {}".format(i, error))
                if error < tol2:
                    break
            
            testlabels = gmm.predict(Xtest)
            trainlabels = gmm.predict(Xtrain)
            testMembs = gmm.predict_proba( Xtest )
            
            return gmm, trainlabels, testlabels, memberships, testMembs
        
        
        
        
        
        def Gmm_Supervised_V3(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = -1,
                           n_clusters = 5, alpha = 0.0001, max_iter = 10, tol = 10**(-3),
                           mix = 0.5 , max_iter2 = 10, tol2 = 10**(-3), lambd = 0.5, 
                           cv = None, calc_tau = None):
            
            """ SUPERVISED PLUS UNSUPERVISED CLUSTERING
                THIS IS THE VERSION  TWO WHICH WE EXPECT TO CONVERGE
            
            
                Xtrain: TRAINING DATA, Default Xtrain
                ytrain: TRAINING TARGET DATA, DEFAULT ytrain
                n_jobs: #JOBS, DEFAULT 2
                n_clusters: #CLUSTERS, DEFAULT:5
                alpha: REGULARIZATION PARAMETER: DEFAULT:0.0001
                max_iter: MAXIMUM NUMBER OF ITERATIONS OF THE SUPERVISED CLUSTERING ALGROITHM
                tol: TOLERANCE OF THE SUPERVISED ALGORITHM
                mix: MIXING COEFICIENT OF THE SUPERVISED AND UNSUPERVISED 
                max_iter2: Maximum times of supervised unsupervised clustering
                tol2: TOLERANCE FOR THE SUPERVISED UNSUPERVISED ALGORITHM
                lambd: MIXING OF OLD WITH NEW MEMBERSHIPS
                """
            
            #TAKE DEFAULT IF NO INPUT   
            if not len(Xtrain):
                Xtrain = self.Xtrain
            
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
                
            if not len(ytest):
                ytest = self.ytest
                
            membershipsTr = np.random.rand(Xtrain.shape[0], n_clusters)
            membershipsTr = ( membershipsTr.T/np.sum( membershipsTr, axis = 1 ) ).T  #normalizating initial memeberships
            
            membershipsTest = np.random.rand(Xtest.shape[0], n_clusters)
            membershipsTest = ( membershipsTest.T/np.sum( membershipsTest, axis = 1 ) ).T
                
            for i in np.arange(max_iter2): #TIMES TO PERFORM THE ALGORITHM
               
                    
               
                labelsSC, membSC, models, _ , _, _= self.supervised_clustering( Xtrain = Xtrain, ytrain = ytrain, 
                                                                        memb_init = membershipsTr, n_clusters = n_clusters,
                                                                        n_jobs = n_jobs, max_iter = max_iter, tol = tol, version = 2, 
                                                                        cv = cv, calc_tau = calc_tau)
                    
                #RUN UNSUPERVISED GMMS WITH INITIAL PARAMETERS GIVEN SUPERVISED TRAINING
                weights, means, covariances_Inv = self.calculateStats( np.concatenate( ( Xtrain, Xtest ), axis = 0 ),
                                                                     np.concatenate( ( membershipsTr, membershipsTest ), axis = 0 ) )
                
                gmm = GaussianMixture(n_components = n_clusters, weights_init = weights, 
                                      means_init = means, precisions_init =  covariances_Inv, 
                                      max_iter = 1).fit( np.concatenate( (Xtrain, Xtest), axis = 0) )
                                      
                membershipsnewTest = gmm.predict_proba( Xtest )
                membershipsnewTr = gmm.predict_proba( Xtrain ) * membSC
                membershipsnewTr = ( membershipsnewTr.T/ np.sum( membershipsnewTr, axis = 1) ).T
                
                errorTr = np.sum( np.abs( membershipsTr - membershipsnewTr) )
                errorTst = np.sum( np.abs( membershipsTest - membershipsnewTest ) )
                error = ( errorTr + errorTst )/( Xtrain.shape[0] + Xtest.shape[0] )
                
                membershipsTr = membershipsnewTr*lambd + membershipsTr*(1-lambd)
                membershipsTest = membershipsnewTest*lambd + membershipsTest*(1-lambd)
                
                print("GMM iteration: {}, error: {}".format(i, error))
                if error < tol2:
                    break
            
            testlabels = np.argmax( membershipsTest, axis = 1 )
            trainlabels = np.argmax( membershipsTr, axis = 1 )
            
            
            
            return models, trainlabels, testlabels, membershipsTr, membershipsTest
        
        
        def Gmm_Supervised_V2(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = -1,
                           n_clusters = 5, alpha = 0.0001, max_iter = 10, tol = 10**(-3),
                           mix = 0.5 , max_iter2 = 10, tol2 = 10**(-3), lambd = 0.5, 
                           cv = None, calc_tau = None):
            
            """ SUPERVISED PLUS UNSUPERVISED CLUSTERING
                THIS IS THE VERSION  TWO WHICH WE EXPECT TO CONVERGE
            
            
                Xtrain: TRAINING DATA, Default Xtrain
                ytrain: TRAINING TARGET DATA, DEFAULT ytrain
                n_jobs: #JOBS, DEFAULT 2
                n_clusters: #CLUSTERS, DEFAULT:5
                alpha: REGULARIZATION PARAMETER: DEFAULT:0.0001
                max_iter: MAXIMUM NUMBER OF ITERATIONS OF THE SUPERVISED CLUSTERING ALGROITHM
                tol: TOLERANCE OF THE SUPERVISED ALGORITHM
                mix: MIXING COEFICIENT OF THE SUPERVISED AND UNSUPERVISED 
                max_iter2: Maximum times of supervised unsupervised clustering
                tol2: TOLERANCE FOR THE SUPERVISED UNSUPERVISED ALGORITHM
                lambd: MIXING OF OLD WITH NEW MEMBERSHIPS
                """
            
            #TAKE DEFAULT IF NO INPUT   
            if not len(Xtrain):
                Xtrain = self.Xtrain
            
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
                
            if not len(ytest):
                ytest = self.ytest
                
            membershipsTr = np.random.rand(Xtrain.shape[0], n_clusters)*10**4
            membershipsTr = ( membershipsTr.T/np.sum( membershipsTr, axis = 1 ) ).T  #normalizating initial memeberships
            
            membershipsTest = np.random.rand(Xtest.shape[0], n_clusters)*10**4
            membershipsTest = ( membershipsTest.T/np.sum( membershipsTest, axis = 1 ) ).T
            
            indexing = np.arange( Xtrain.shape[0] )
            fs = np.zeros( [ Xtrain.shape[0], n_clusters ] ) #ERROR MATRIX TO UPDATE MEMBERSHIPS
            
                
            for i in np.arange( max_iter2 ): #TIMES TO PERFORM THE ALGORITHM
               
                    
                models = []
                for  j in np.arange( n_clusters ):
                   # sgd, _ = self.logistic_regr(Xtrain = Xtrain, ytrain = ytrain, cv = cv, calc_tau = calc_tau,
                                              # sample_weight = membershipsTr[:, j] , alpha = alpha)
                    
                    sgd, _ = self.logistic_regrReal(Xtrain = Xtrain, ytrain = ytrain, cv = cv, calc_tau = calc_tau,
                                                sample_weight = membershipsTr[:, j] , alpha = alpha)
                    
                    proba = sgd.predict_proba( Xtrain )
                    fs[:,j] = proba[indexing, ytrain]
                    models.append( sgd )
                    
                #RUN UNSUPERVISED GMMS WITH INITIAL PARAMETERS GIVEN SUPERVISED TRAINING
                weights, means, covariances_Inv = self.calculateStats( np.concatenate( ( Xtrain, Xtest ), axis = 0 ),
                                                                     np.concatenate( ( membershipsTr, membershipsTest ), axis = 0 ) )
                
                gmm = GaussianMixture(n_components = n_clusters, weights_init = weights, 
                                      means_init = means, precisions_init =  covariances_Inv, 
                                      max_iter = 1).fit( np.concatenate( (Xtrain, Xtest), axis = 0) )
                                      
                membershipsnewTest = gmm.predict_proba( Xtest )
                membershipsnewTr = gmm.predict_proba( Xtrain ) * fs
                membershipsnewTr = ( membershipsnewTr.T/ np.sum( membershipsnewTr, axis = 1) ).T
                
                errorTr = np.sum( np.abs( membershipsTr - membershipsnewTr) )
                errorTst = np.sum( np.abs( membershipsTest - membershipsnewTest ) )
                error = ( errorTr + errorTst )/( Xtrain.shape[0] + Xtest.shape[0] )
                
                membershipsTr = membershipsnewTr*lambd + membershipsTr*(1-lambd)
                membershipsTest = membershipsnewTest*lambd + membershipsTest*(1-lambd)
                
                print("/n/n/n GMM iteration: {}, error: {}/n/n/n".format(i, error))
                if error < tol2:
                    break
            
            testlabels = np.argmax( membershipsTest, axis = 1 )
            trainlabels = np.argmax( membershipsTr, axis = 1 )
            
            
            
            return models, trainlabels, testlabels, membershipsTr, membershipsTest
        
       
        #PERFORM LOGISTIC REGRESSION INTO THE WHOLE DATASET
        def logistic_all(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = -1, alpha = 0.0001 ,
                         cv = None, calc_tau = None):
            """ PERFORMS LOGISTIC REGRESSION TO TRAIN AND TEST DATA 
                Xtrain: XTRAIN DATA, DEFAULT, Xtrain
                ytrain: TARGET TRAINING DATA, DEFAULT: ytrain
                n_jobs: #JOBS
                alpha: L1 REGULARIZATION PARAMETER, DEFAULT: 0.0001
            """
            
            
            if not len(Xtrain):
                Xtrain = self.Xtrain
            
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
            
            if not len(ytest):
                ytest = self.ytest
            
            weights = []   # matrix for logistic regression weights for each cluster
            metrics = []   # matrix with metrics for Training test
            metricsTs = [] # matrix with metrics for test set
            columns = ['cluster', 'precision', 'recall', 'accuraccy', 'balanced_accuraccy', 'f1', 'auc']
            
            sgd, tau = self.logistic_regr(Xtrain, ytrain, n_jobs = n_jobs, alpha = alpha, cv = cv, calc_tau = calc_tau)  #TRAINING
            
            metLog, _ = self.calc_metrics( classifier =  sgd, X = Xtrain, y =  ytrain, tau = tau )#CALCULATE METRICS FOR TRAINING
            weights.append( sgd.best_estimator_.coef_)
            metrics.append(metLog)
            
            metLogtest, _ =self.calc_metrics( classifier = sgd, X = Xtest, y = ytest, tau  = tau ) #CALCULATE METRICS FOR TESTNG DATA
           
             
            metricsTs.append( metLogtest )
            
            metrics = pd.DataFrame( metrics, columns = columns )
            metricsTs = pd.DataFrame( metricsTs, columns = columns )
            
            positiveTrain = len( np.where( ytrain == 1)[0] )
            positiveTest = len( np.where(ytest == 1)[0] )
            
            col_names =[ 'Train_Posi#', 'Test_Posi#', 'Train_Tot#', 'Test_Tot#' ]
            posit = np.array( [ positiveTrain, positiveTest, ytrain.shape[0], ytest.shape[0] ] )
           # posit = pd.Dataframe(posit, columns = col_names)
            
            return sgd,  metrics, metricsTs, weights, tau
        
        
        
        def Forest_all(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = -1, alpha = 0.0001 ,
                         cv = None, calc_tau = None):
            """ PERFORMS Random Forest TO TRAIN AND TEST DATA 
                Xtrain: XTRAIN DATA, DEFAULT, Xtrain
                ytrain: TARGET TRAINING DATA, DEFAULT: ytrain
                n_jobs: #JOBS
                alpha: L1 REGULARIZATION PARAMETER, DEFAULT: 0.0001
            """
            
            
            if not len(Xtrain):
                Xtrain = self.Xtrain
            
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
            
            if not len(ytest):
                ytest = self.ytest
            
            weights = []   # matrix for logistic regression weights for each cluster
            metrics = []   # matrix with metrics for Training test
            metricsTs = [] # matrix with metrics for test set
            columns = ['cluster', 'precision', 'recall', 'accuraccy', 'balanced_accuraccy', 'f1', 'auc']
            
            sgd  =  RandomForestClassifier(n_estimators= 200, max_depth= None,
                              random_state=0)
            sgd = sgd.fit(Xtrain, ytrain)
            
            
            tau = self.optimalTau(sgd.predict_proba(Xtrain)[:, 1], ytrain )
            
            metLog, _ = self.calc_metrics( classifier =  sgd, X = Xtrain, y =  ytrain, tau = tau )#CALCULATE METRICS FOR TRAINING
         #   weights.append( sgd.best_estimator_.coef_)
            metrics.append(metLog)
            
            metLogtest, _ =self.calc_metrics( classifier = sgd, X = Xtest, y = ytest, tau  = tau ) #CALCULATE METRICS FOR TESTNG DATA
           
             
            metricsTs.append( metLogtest )
            
            metrics = pd.DataFrame( metrics, columns = columns )
            metricsTs = pd.DataFrame( metricsTs, columns = columns )
            
          #  positiveTrain = len( np.where( ytrain == 1)[0] )
            #positiveTest = len( np.where(ytest == 1)[0] )
            
         #   col_names =[ 'Train_Posi#', 'Test_Posi#', 'Train_Tot#', 'Test_Tot#' ]
         #   posit = np.array( [ positiveTrain, positiveTest, ytrain.shape[0], ytest.shape[0] ] )
           # posit = pd.Dataframe(posit, columns = col_names)
            
            return sgd,  metrics, metricsTs,  tau
        
        def Neural_Network(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = -1, alpha = 0.0001 ,
                         cv = None, calc_tau = None):
            """ PERFORMS Random Forest TO TRAIN AND TEST DATA 
                Xtrain: XTRAIN DATA, DEFAULT, Xtrain
                ytrain: TARGET TRAINING DATA, DEFAULT: ytrain
                n_jobs: #JOBS
                alpha: L1 REGULARIZATION PARAMETER, DEFAULT: 0.0001
            """
            
            
            if not len(Xtrain):
                Xtrain = self.Xtrain
            
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
            
            if not len(ytest):
                ytest = self.ytest
            
          # weights = []   # matrix for logistic regression weights for each cluster
            metrics = []   # matrix with metrics for Training test
            metricsTs = [] # matrix with metrics for test set
            columns = ['cluster', 'precision', 'recall', 'accuraccy', 'balanced_accuraccy', 'f1', 'auc']
            
            sgd  =  MLPClassifier( hidden_layer_sizes = (10, 5, 5), early_stopping = True)
            sgd = sgd.fit(Xtrain, ytrain)
           
            
            tau = self.optimalTau(sgd.predict_proba(Xtrain)[:, 1], ytrain )
            
            metLog, _ = self.calc_metrics( classifier =  sgd, X = Xtrain, y =  ytrain, tau = tau )#CALCULATE METRICS FOR TRAINING
         #   weights.append( sgd.best_estimator_.coef_)
            metrics.append(metLog)
            
            metLogtest, _ =self.calc_metrics( classifier = sgd, X = Xtest, y = ytest, tau  = tau ) #CALCULATE METRICS FOR TESTNG DATA
           
             
            metricsTs.append( metLogtest )
            
            metrics = pd.DataFrame( metrics, columns = columns )
            metricsTs = pd.DataFrame( metricsTs, columns = columns )
            
          #  positiveTrain = len( np.where( ytrain == 1)[0] )
            #positiveTest = len( np.where(ytest == 1)[0] )
            
         #   col_names =[ 'Train_Posi#', 'Test_Posi#', 'Train_Tot#', 'Test_Tot#' ]
         #   posit = np.array( [ positiveTrain, positiveTest, ytrain.shape[0], ytest.shape[0] ] )
           # posit = pd.Dataframe(posit, columns = col_names)
            
            return sgd,  metrics, metricsTs,  tau
        
        #<<<<<<<ANALYSIS FUNCTIONS>>>>>>>>
        
        #PURE K MEANS ANALYSIS
       
        
        def kmeansAnalysis_1(self,  Xtrain = [],  Xtest = [], ytrain = [],  ytest = [], n_jobs = 2, n_clusters = 5):
            
            """HIGH COST ANALYSIS AND STATISTICS 
            FOR THE SPARCS DATASET
            Xtrain: TRAIN DATA, DEFAULT CLASSES Xtrain
            ytrain: TEST TARGET DATA, DEFAULT CLASSES ytrain 
            n_jobs: #JOBS , DEFAULT :2
            n_clusters: #CLUSTERS, DEFAULT: 5
            
            """
            
            if not len(Xtrain):
                Xtrain = self.Xtrain
                
            if not len(Xtest):
                Xtest = self.Xtest
                
            if not len(ytrain):
                ytrain = self.ytrain
                
            if not len(ytest):
                ytest = self.ytest
            
                
            kmeans  = self.k_means(Xtrain = Xtrain, n_jobs = n_jobs, n_clusters = n_clusters) #perform kmeans
            labels = kmeans.labels_
            
            
            #TO DO CALCULATE STATS FOR TESTING  DATA TOO
            
            
            highStats, lowStats = self.getClusterStats( ytrain, labels, n_clusters )
            
            return highStats, lowStats, kmeans
        
        
        
        
        
        
        
        #<<<<<<<   HELPER FUNCTIONS   >>>>>>>> 
        #SPECIFIC TO SPARCS DATASETS (OR WHATEVER ANALYSIS WITH HIGH COST AND LOW COST DATAPOINTS)
        
        
        
        def getClusterStats(self, y =  [], labels = [], n_clusters = []):
            """FOR EACH CLUSTER RETURNS THE SOME STATISTICS OF THE CLUSTER 
                ytrain: TARGET TRAINING DATA, DEFAULT: ytrain OF THE CLASS
                labels: LABELS DEPICTING EACH DATAPOINT"S CLASS 0-N_CLUSTERS
                n_clusters: NUMBER OF CLUSTERS
                
                RETURNS TWO DATAFRAMES WITH HIGH COST AND LOW COST STATISTICS RESPECTIVELY
            """
            if not len(y):
                y = self.ytrain
                
            colsH = ['Cluster', 'High_Cost_%','High_Cost_tot', 'Cluster_total', 'Cluster_pop_%','High_tot_Perc%' ]
            colsL = ['Cluster', 'Low_Cost_%','Low_Cost_tot', 'Cluster_total', 'Cluster_pop_%','Low_tot_Perc%']
            
            
            high = []
            low = []
            
            for cluster in np.arange(n_clusters):  #iterating over all clusters
                
                h, l = self.cluster_statistics(cluster, labels, y)
                
                high.append(h)
                low.append(l)
            
            high = pd.DataFrame(high, columns = colsH)
            low =  pd.DataFrame(low, columns = colsL)
            
            return high, low
        
        
        
         
        def cluster_statistics(self, cluster_number, labels, y):
                """ 
                Takes the cluster number, the labels of the data points (in which cluster each point belongs)
                and the ytrain and calculates some usefull statistics for the high cost case
                """
                totalHigh = len(np.where(y == 1)[0])               #TOTAL NUMBER OF HIGH COST PATIENTS IN THE TRAINING DATASET
                totalLow =  len(np.where(y == 0)[0])               #TOTAL NUMBER OF LOW COST PATIENTS IN THE  TRAINING DATASET
                clusterIndex = np.where( labels == cluster_number)[0]   #indexes belonging to cluster "cluster"
                highcost = np.where( y[clusterIndex] == 1)[0]      #indexes of high cost patients within cluster "cluster"
                highPerc = len(highcost)/len(clusterIndex)              #percentage of high cost patients in the class
                highCostTot = len(highcost)                             #NUMBER OF HIGH COST PATIENTS IN THE CLASS                                                    
                ClusterTot = len(clusterIndex)                          #NUMBER OF PATIENTS WITHIN THE CLUSTER "cluster"
                ClusterPerc = len(clusterIndex)/y.shape[0]         #PERCENTAGE OF PATIENTS IN THE CLUSTER VERSUS WHOLE POPULATION OF TRAINING DATA
                highTot = len(highcost)/totalHigh                       #PERCENTAGE OF HIGH COST PATIENTS IN THE CLUSTER VERSUS ALL PATIENTS IN THE TRAINIBG DATA
                lowTot = (len(clusterIndex) - highTot)/totalLow         #PERCENTAGE OF LOW COST PATIENTS IN THE CLUSTER VERSUS ALL PATIENTS IN THE TRAINIBG DATA
               
                high = [cluster_number, highPerc, highCostTot, ClusterTot, ClusterPerc, highTot]
                low =  [cluster_number, 1-highPerc, ClusterTot - highCostTot, ClusterTot, ClusterPerc,lowTot]
                return high, low
            
        
        
        
        
        
            
        #HELPER FUNCTION FOR SUPERVISED CLUSTERING
        #TO INITIALIZE THE MEMBERSHIPS
        def initializeMember(self,labels, n_clusters):
            """ HELPER FUNCTION FOR SUPERVISED CLUSTERING"""
            
            Xdim = labels.shape[0]
            memberships = np.zeros([Xdim,  n_clusters])
    
            for i in np.arange(Xdim):
                memberships[i, labels[i]] = 1
    
            return memberships  
        
        
        
        
        
        #CALCULATES THE WEIGHTS, MEANS AND COVARINACES MATRICES THAT WILL ACT AS AN INITIALIZER TO UNSUPERVISED 
        #CLUSTERING WITH GAUSSIAN MIXTURES MODELS
        def calculateStats(self, X, memberships):
            """ HELPER TO SUPERVISE CLUSTERING TO CALCULATE THE MEAN< WEIGHTS AND COVARIANCES
            AS AN INITIAL STEP TO USE THEM AS INITIALIZERS TO GMM's"""
            
            N = X.shape[0]
            weights = np.sum(memberships, axis = 0)/N
            means = np.zeros([X.shape[1],memberships.shape[1]])
            cov = np.zeros([memberships.shape[1], X.shape[1], X.shape[1]]) #3D array containing the memberships
            
            for  i  in np.arange( memberships.shape[1] ):
       
                Ni = weights[i]*N
                means[:,i] = np.sum( (X.T*memberships[:,i]).T, axis = 0 ) /(Ni)
                cov[i,:,:] = ((memberships[:,i])*X.T@X)/Ni - np.expand_dims(means[:,i], axis = 1)@np.expand_dims(means[:,i], axis = 1).T +np.eye(X.shape[1])*10**(-4)
                cov[i, :, :] = np.linalg.inv( cov[i, :, :] ) 
        
            return weights, means.T, cov
        
        
        
        
        
        #TAKES A CLASSIFIER X DATA Y DATA AND RETURNS METRICS ON THEM
        def calc_metrics(self, classifier = [], X = [], y = [], cluster = -1, tau = None, custom_prob = []):
             """ computes metrics such us accuracy, precision, recall, balanced accuracy """
             
             if  (tau is None ) :  #case Do not calculate Tau
                 
                 if not len(custom_prob): #case we dont give custom probabilities
                     predictions = classifier.predict_proba( X )[:, 1]  #PREDICT THE LABELS OF X DATA ACCORDING TO THE TRAINED CLASSIFIER
                     
                 else:
                     predictions = custom_prob
                     
                 auc = roc_auc_score( y , predictions )
                 roc = roc_curve(y, predictions)
                 predictions[ np.where( predictions >= 0.5)[0] ] = 1
                 predictions[ np.where( predictions < 0.5)[0] ] = 0
                 
             
             else: #when we give a tau
                 
                 if not len(custom_prob): #case when we dont give custom probabilities
                     
                     probabilities = classifier.predict_proba( X )[ :, 1 ]
                     
                 else:
                     probabilities = custom_prob
                     print( 'I am here cistum Prob with TAU {}'.format(custom_prob.shape))
                    
                 auc = roc_auc_score( y , probabilities)  
                 roc = roc_curve(y, probabilities)
                 probabilities[ np.where( probabilities >= tau ) ] = 1
                 probabilities[ np.where( probabilities < tau ) ] = 0
                 predictions = probabilities
                 
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
         
            
            
         
            
        def addTotalBalanced(self, CMetrics,  CWeights):
            """ A helper function that adds the total balanced accuracy
                to the metrics  matrix  created for the cclustering methods """
                
            data = CMetrics.values
            columns = CMetrics.columns.tolist()
            columns.append('Total_BAcc')
            columns.append('Total_F1')
            columns.append('Total_auc')
            
            totAcc = np.sum( data[:,-3]*CWeights )    #total Balanced accuracy
            totF1 = np.sum( data[:,-2] * CWeights )   #total F1 score
            totAuc = np.sum( data[:,-1] * CWeights )   #total F1 score
            totMat = np.ones([CWeights.shape[0], 1] )*totAcc
            totMat1 = np.ones([CWeights.shape[0], 1] )*totF1
            totMat2 = np.ones([CWeights.shape[0], 1] )*totAuc
            
            data = np.concatenate( ( data, totMat, totMat1, totMat2 ), axis = 1)
            
            data = pd.DataFrame(data, columns = columns)
            
            return data
        
        
        def CreateCloudsClustersWeights( self, cols, topNumber, MethodName, n_clusters, weights):
            
            cols = np.array( cols )
            for i in np.arange( n_clusters ):
                weightsCluster = np.squeeze( weights[i], axis = 0 )
                weightsPosInd = np.where( weightsCluster > 0 )[ 0 ]
                weightsNegInd = np.where( weightsCluster < 0 )[ 0 ]
                colsPos = cols[weightsPosInd]
                colsNeg = cols[weightsNegInd]
                weightsPos = weightsCluster[ weightsPosInd ]
                weightsNeg = np.abs( weightsCluster[ weightsNegInd ] )
               # print(weightsPos.shape, weightsNeg.shape )
            
                if len(weightsPosInd) < topNumber:
                     posIndsort = np.flip( np.argsort( weightsPos ), axis = 0)[0: len(weightsPosInd) ]
                else:
                     posIndsort = np.flip( np.argsort( weightsPos ), axis = 0)[0: topNumber]
                
                if len( weightsNegInd ) < topNumber:
                     negIndsort = np.flip( np.argsort( weightsNeg ), axis = 0)[0: len( weightsNegInd ) ]
                else:
                     negIndsort = np.flip( np.argsort( weightsNeg ), axis = 0)[0: topNumber ]
                
                
               
                if len( weightsPosInd ):
                    
                    #textPos = self.generateText( colsPos[posIndsort], np.ceil( weightsPos[ posIndsort ] ) )
                    textPos = dict( zip( colsPos[posIndsort], weightsPos[ posIndsort ]))
                    namePos = MethodName + 'WeightsPosCluster{}.png'.format(i)
                    
                    wordcloud =WordCloud(background_color="white", collocations = False, stopwords = STOPWORDS, 
                                   min_font_size = 7, margin = 0).generate_from_frequencies( textPos )
                    fig, ax = plt.subplots(nrows = 1, ncols = 1)
                    fig.set_size_inches(12,12)
                    ax.imshow(wordcloud, interpolation = 'bilinear')
                    ax.axis('off')
                    fig.savefig('wordcloudsWeights/'+ namePos, format = 'png',  dpi = 300)
                else:
                    
                    textPos = []
                    
                if len( weightsNegInd ):
                    
                    #textNeg = self.generateText( colsNeg[negIndsort], np.ceil( weightsNeg[ negIndsort ] ) )    
                    textNeg = dict( zip( colsNeg[ negIndsort], weightsNeg[ negIndsort ]))
                    nameNeg = MethodName + 'WeightsNegCluster{}.png'.format(i)
                   
                    
                    wordcloud = WordCloud(background_color="white", collocations = False, stopwords = STOPWORDS, 
                                   min_font_size = 7, margin = 0).generate_from_frequencies(textNeg)
                    fig, ax = plt.subplots(nrows = 1, ncols = 1)
                    fig.set_size_inches(12,12)
                    ax.imshow(wordcloud, interpolation = 'bilinear')
                    ax.axis('off')
                    fig.savefig('wordcloudsWeights/'+ nameNeg, format = 'png',  dpi = 300)
                    
                else:
                    
                    textNeg = []
                
                
                
                
            
            return textPos, textNeg
        
        def CreateCloudsClusters(self, data, cols, topNumber,  MethodName, labels, n_clusters):
            
            tfIdf = self.calculateTFIDF2( data, labels, n_clusters )
            
            for i in np.arange( n_clusters ):
                name = MethodName + 'Cluster{}.png'.format(i)
                nameList = self.CreateClouds( data[ np.where(labels == i)[0] ], cols, topNumber, name, tfIdf[i],'{}'.format(i))
            
            return nameList, tfIdf
            
        
        def CreateClouds(self, data, cols, topNumber, name, tfIdf, cN):
            
            namesList, counts, indexes, columns = self.makeWordCloud2(data, cols, topNumber, tfIdf)
            dictio = dict(zip(columns, counts))
            if not namesList:
                return []
            wordcloud = WordCloud(background_color="white", collocations = False, stopwords = STOPWORDS, 
                                   min_font_size = 7, margin = 0).generate_from_frequencies(dictio)#.to_file('wordclouds/'+ name)
            fig, ax = plt.subplots(nrows = 1, ncols = 1)
            fig.set_size_inches(12,12)
            ax.imshow(wordcloud, interpolation = 'bilinear')
            ax.axis('off')
            fig.savefig('wordclouds/'+ name, format = 'png',  dpi= 300 )
          #  np.save('wordclouds/'+'nameList'+cN+'.npy', namesList)
            return dictio
            
            
        
        def makeWordCloud (self, data, cols, topNumber, tfIdf ):
    
            indexes, columns = self.bC ( data, cols)
            if tfIdf is not None:
                
                counts = tfIdf[indexes]
                maxCounts =  np.flip( np.argsort( counts ), axis = 0 ) [0:topNumber ]
            
            else:
                counts = self.countBinary( data[:, indexes] ) 
                maxCounts = np.flip( np.argsort( counts ), axis = 0 )[ 0:topNumber ]
            
            namesList = []
    
            for i in  maxCounts:
                #print(i)
                helpMe = [cols[int(i)]]* int(counts[int(i)])
                namesList.extend(helpMe)
    
            names = ' '.join(namesList)    
            
            names = self.generateText( np.array( cols )[maxCounts], counts[ maxCounts ])
            return names, counts, indexes, columns
        
        def makeWordCloud2 (self, data, cols, topNumber, tfIdf ):
    
            indexes, columns = self.bC ( data, cols)
            if tfIdf is not None:
                
                counts = tfIdf[indexes]
                maxCounts =  np.flip( np.argsort( counts ), axis = 0 ) [0:topNumber ]
            
            else:
                counts = self.countBinary( data[:, indexes] ) 
                maxCounts = np.flip( np.argsort( counts ), axis = 0 )[ 0:topNumber ]
            
            namesList = []
    
            for i in  maxCounts:
                #print(i)
                helpMe = [cols[int(i)]]* int(counts[int(i)])
                namesList.extend(helpMe)
    
            names = ' '.join(namesList)    
            
            names = self.generateText( np.array( cols )[maxCounts], counts[ maxCounts ])
            return names,  counts, indexes, columns
    
        def generateText(self, ColumnsNames, ColumnsCounts):
            
            namesList = []
            for i  in np.arange( len(ColumnsCounts) ):
                helpMe = [ColumnsNames[int(i)]]* int(ColumnsCounts[int(i)])
                namesList.extend(helpMe)
            
            names = ' '.join(namesList)
            
            return names
            
                
        
        def countBinary(self, data ):
            counts = np.sum( data, axis = 0 )
    
            return counts
   
   

        def bC(self, data, cols) :
    
            indexes = []
            columns = []
    
            for i, cols in enumerate(cols[ : -1 ]):
               # print(cols, i)
                if np.isin( data[:,i], [0,1] ).all():
                    indexes.append(i)
                    columns.append(cols)
        
            return indexes, columns
        
        
        def calculateTFIDF(self,  data, labels, n_clusters ):
            
            tfIdf = []
            idf = np.zeros( data.shape[1] )     
            
            for i in np.arange( n_clusters ):
                
                index = np.where( labels == i )[0]
                count = np.sum( data[index], axis = 0 )
                indexNonZeros = np.where( count != 0)[0]
                idf[indexNonZeros] = idf[ indexNonZeros ] + 1
                tfIdf.append( count )
            
            idf[ np.where(idf == 0) ] = n_clusters  #this is needed when we do not use all the data
            for i in np.arange( n_clusters ):
                
                tfIdf[i] = tfIdf[i]  * np.log( n_clusters / (idf ) )
                
                
            return tfIdf
        
        def calculateTFIDF2( self, data, labels, n_clusters ):
            tfIdf = []
            idf = np.zeros( data.shape[1] )
            dataCount = np.sum( data, axis = 0)
            
            for  i in np.arange( n_clusters ):
                index = np.where( labels == i )[0]
                count = np.sum( data[index], axis = 0 )
                clusterWeight = len(index)/data.shape[0]  #account the Clusters percentage in data Points
               # totCount = np.sum(count)/ ( 2 * data.shape[1] )
                
                for j in np.arange( data.shape[1] ):
                    
                    if count[j] > (dataCount[ j ] * clusterWeight) :
                        idf[j] += 1
                tfIdf.append( count )
                
            idf[ np.where(idf == 0) ] = n_clusters  #this is needed  when  we do not use all the data
            for i in np.arange( n_clusters ):
                tfIdf[i] = np.ceil(tfIdf[i]  * np.log( n_clusters / ( idf ) ))
                
            print(max(idf), n_clusters)    
            return tfIdf
                
            
            
        
        
        def optimalTau(self, probabilities, ylabels):
            
            """ Finds the Optimal tau based on the F1 score"""
            
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
                    prec = precision
                    rec = recall
                    
            
            return threshold #, f1, np.array(prob_F1), prec, rec
                
                
                
            
        
        
        
        
            
            
                
            
            
            
        
        
        
            
            
        
            