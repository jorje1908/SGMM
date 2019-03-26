#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:14:37 2019

@author: george
"""

import numpy as np
#import pandas as pd
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
#from sklearn.metrics import precision_score, accuracy_score, recall_score, \
#balanced_accuracy_score, f1_score
#from sklearn.mixture import GaussianMixture
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import roc_auc_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClas
from  scipy.stats import multivariate_normal


####THIS CODE HAS NUMERICAL ISSUES AT THE SPARCS DATASET

class SupervisedGMM():
    
    """ 
        THIS CLASS IMPLEMENTS THE SUPERVISED GMM ALGORITHM 
        IT USES SOME PARTS OF SCIKIT LEARN TO ACCOMPLISH THIS    
    """
    
    
    def __init__(self, max_iter = 100, cv = 2, mix = 0.5, Cs = [1000], 
                 max_iter2 = 10, penalty = 'l1', scoring = 'f1',
                 solver = 'saga', n_clusters = 2, tol = 10**(-3 ) ):
        
        
        
        """ LOGISTIC REGRESSION PARAMETERES:
            
            max_iter:[INT] #Iterations of Opt Algorithm: DEF: 100
            cv:[INT] Cross Validation: Default 3 Fold
            Cs: [LIST] Inverse of Regularization Parameter: DEF: 1000
            penlaty:[FLOAT] Regularization
            solver: [STRING] DEF: 'saga', Solvers used by scikit learn
            for logistic Regression 
            scoring:[STRING] score to optimize in cross validation: DEF: 'f1'
            
            GMMS PARAMETERES:
            mix: In what Percentages to Soft Cluster Memberships: DEF: 0.5   
            max_iter2: Maximum # of EM Iterations, DEF: 10 
            n_clusters: #of Soft Clusters: DEF: 2
        """
        
        
        self._max_iter = max_iter
        self._cv = cv
        self._mix = mix
        self._Cs = Cs
        self._max_iter2 =  max_iter2
        self._penalty = penalty
        self._scoring = scoring
        self._solver = solver
        self._n_clusters = n_clusters
        self._tol = tol
        #self.Xtrain = []
        #self.Xtest = []
        #self.ytrain = []
        #self.ytest = []
        
        #THE FOLLOWING ATTRIBUTES ARE SETTED AFTER FITTING THE ALGORITHM
        #TO DATA
        self.Gmms = None  #this exists only when we fit the model a list
                         #with fitted Gaussians one for each class
        self.mixes = None  #Gmms mixes
        self.LogRegr = None #Logistic Regression Models
        self.params = None #parameteres returned by the gmmModels
        self.fitParams = None #parameters of the final model
                            #membership matrices for test and train data
                            #hard clustering labels of test and train data
        self.mTrain = None
        self.mTest = None
        
    def fit(self, Xtrain = None, ytrain = None, Xtest = None):
        
        #CHECK IF ALL DATA ARE GIVEN
        if Xtrain is None or ytrain is None or Xtest is None :
            print(" Please Give Xtrain, ytrain, Xtest and ytest data ")
            return
        
        #PARAMETERES TO BE USED BY THE FIT FUNCTION
        n_clusters = self._n_clusters
        max_iter = self._max_iter
        cv = self._cv
        mix = self._mix
        penalty = self._penalty
        scoring = self._scoring
        solver = self._solver
        max_iter2 = self._max_iter2
        dimXtrain = Xtrain.shape[0]
        dimXtest = Xtest.shape[0]
        Cs = self._Cs
        tol = self._tol
        
        
        
        #INITIALIZE MEMBERSHIP FUNCTIONS
        #WE KEEP SEPARATED TRAIN AND TEST MEMBERSHIPS
        #BECAUSE TRAIN IS SUPERVISED MEMBERSHIP
        #TEST IS UNSUPERVISED
        mTrain = np.random.rand( dimXtrain, n_clusters)
        mTest = np.random.rand( dimXtest, n_clusters )


        #NORMALIZE MEMBERSHIPS SO EACH ROW SUMS TO 1
        sumTrain = np.sum( mTrain, axis = 1)
        sumTest = np.sum( mTest, axis = 1 )
        mTrain = ( mTrain.T / sumTrain ).T
        mTest = ( mTest.T / sumTest ).T    
        
        #print(np.sum(mTrain, axis = 1), np.sum(mTest, axis = 1))
        
        #SET SOME NEEDED PARAMETERES
        #FOR USE IN THE FOR LOOP
        indexing = np.arange( dimXtrain )
        logiProb = np.zeros([dimXtrain, n_clusters])
       
       
        
        #START FITTING ALGORITHM
        
        for iter2 in np.arange( max_iter2 ):
            #FITING THE L1 LOGISTIC REGRESSIONS
            models = [] #EVERY IETARTION CHANGE MODELS 
                        # IN ORDER TO TAKE THE MODELS OF LAST ITERATION
                        #OR THE MODELS BEFORE ERROR GO BELOW TOLERANCE
            for clust in np.arange( n_clusters ):
                                                                 
                #FIT THE L! LOGISTIC REGRESSION MODEL
                #CROSS VALIDATION MAXIMIZING BE DEFAULT THE F1 SCORE
                model = LogisticRegressionCV(Cs = Cs, penalty = penalty,
                             scoring = scoring, solver = solver, max_iter =
                             max_iter,cv = cv).fit( Xtrain, ytrain,
                             mTrain[:, clust] )
                
                #FOR EACH CLUSTER APPEND THE MODEL in MODELS
                models.append( model )  
               
                #PREDICT PROBABILITIES FOR BEING IN CLASS 1 or 0
                #FOR THE TRAINING DATA
                proba = model.predict_proba( Xtrain )
                
                #FOR EACH DATA POINT WE TAKE THE PROBABILITY 
                # OF BEING IN THE CORRECT CLASS
                #FOR EXAMPLE IF x1 BELONGS IN CLASS 1
                # WE TAKE THE PROBABILITY PREDICTED BY THE CLUSTER 
                # SPECIFIC MODEL OF BEING IN CLASS 1
                #IF THIS IS HIGH IT WILL ENHANCE THE PROB FOR THE POINT TO BE
                #IN THE CLUSTER...
                logiProb[:, clust]  = proba[ indexing, ytrain ] 
            
            #print(logiProb[0,:])
            #WE TAKE THE MEMBERSHIPS AND ALL THE DATA
            #TO FIT THE GAUSSIANS USING THE EM ALGORITHM FOR GMM 
            data = np.concatenate( ( Xtrain, Xtest ), axis = 0 )
            mAll = np.concatenate( (mTrain, mTest ), axis = 0 )
            
            #params is  a dictionary with the following structure
            #params = {'cov':cov, 'means': means, 'pis' : pis, 
            #                 'probMat':probMat, 'Gmms': Gmms}
            #cov: list of covariances
            #means: list of means
            #pis : list of probabilities of a specificc gaussian to be chosen
            #probMat: posterior probability membership matrix
            #Gmms ; a list of Object with the Gaussians for each class
            params = self.gmmModels( data, mAll )
                
            gmmProb = params['probMat']
            #SOME DEBUGGING
            #print(gmmProb)
           # if iter2 == 1:
             #return params
                
            #CALCULATE NEW MEMBERSHIPS FOR TRAIN AND TEST
            mNewTest = gmmProb[dimXtrain :, :]
            mNewTrain = logiProb * gmmProb[0: dimXtrain, :]
                
            #NORMALIZE NEWMEMBERSHIPS
            sumTrain = np.sum( mNewTrain, axis = 1)
            sumTest = np.sum( mNewTest, axis = 1 )
           # print(sumTrain, sumTest)
            mNewTrain = ( mNewTrain.T / sumTrain ).T
            mNewTest = ( mNewTest.T / sumTest ).T  
                
                
            #EVALUATE ERROR
            errorTr = np.sum( np.abs( mTrain - mNewTrain) )
            errorTst = np.sum( np.abs( mTest - mNewTest ) )
            error = ( errorTr + errorTst )/( dimXtrain + dimXtest )
            
            #MAKE A SOFT CHANGE IN MEMEBRSHIPS MIXING OLD WITH NEW 
            # MEMBERSHIPS WITH DEFAULT MIXING OF 0.5
            mTrain = mNewTrain*mix + mTrain*(1-mix)
            mTest = mNewTest*mix + mTest*(1-mix)
            
           # print( np.sum(mTrain, axis = 1))
                
            #SETTING ALL MODELS AS ATTRIBUTES OF THE CLASS
            #SO WE CAN USE OUTSIDE OF THE CLASS TOO IF WE WANT FOR PREDICTION
            #THIS MODEL ASSUMES THAT WE HAVE THE TEST DATA AND WE JUST
            #DO NOT KNOW THE LABELS BECAUSE IT USES BOTH TRAIN AND TEST 
            #IN CLUSTERING
            self.Gmms = params['Gmms']
            self.mixes = params['pis']
            self.LogRegr = models
            self.params = params
                
            print("GMM iteration: {}, error: {}".format(iter2, error))
            if error < tol:
                 break
                
        #TAKING HARD CLUSTERS IN CASE WE WANT TO USE LATER       
        testlabels = np.argmax( mTest, axis = 1 )
        trainlabels = np.argmax( mTrain, axis = 1 )
        
        fitParams = {'mTrain' : mTrain, 'mTest': mTest, 'labTest': testlabels,
                     'labTrain' : trainlabels }
        self.mTrain = mTrain
        self.mTest = mTest
        self.fitParams = fitParams
        
        return self
            
            
                
                
                
     #FITTING THE GAUSSIAN MIXTURE MODEL
             
    def gmmModels(self, X, members ):
            
            """
                Calculates the Mixtures of Gaussians Parameters
                Calculates the Mixtures of Gaussians in the form of a list
                of objects of Gaussians for each cluster
                
                X : Train and Test data together
                members: Posterior Probabibilities for each cluster
                             and each data point (memberships)
                             
                Returns: a list with the covariances matrices of the Gaussians,
                a list with the mixing parameteres,
                a list with the means of the gaussians,
                the probability matrix with the posteriors for each data
                point and each cluster,
                a list with the Gaussians as Object
                All these it returns in the form of a dictionary
                
            """
                
            clusters = members.shape[1]
            cov = [] #list with covariance matrices
            means = [] #list of means
            pis = [] #list of mixing coefficients
            probMat = np.zeros( [X.shape[0], clusters] )
            Gmms = []
            logprobaMatrix = np.zeros([X.shape[0], clusters])
                    
            for cl in np.arange( clusters ):
               
                # FOR EACH CLUSTER USE THE EM ALGORITHM
                # TO CREATE THE NEW MEMBERSHIP MATRIX OF THE GAUSSIANS
                #IT IS NOT EXACTLY THE MEMBERSHIP BECAUSE IT IS
                # NORMALIZED  AFTER THIS FUNCTION ENDS
                covCl, mCl, piCl, logproba, model = self.calcGmmPar( X, 
                                                                members[:,cl]) 
                   
                logprobaMatrix[:,cl] = logproba 
                
                cov.append( covCl )
                means.append( mCl )
                pis.append( piCl )
                Gmms.append( model )
                
           
            
            
            maxLog = np.max(logprobaMatrix, axis = 1)
            logprobaMatrix = ( logprobaMatrix.T - maxLog).T
            probMat = np.exp( logprobaMatrix )
            sumRel = np.sum( probMat, axis = 1)
            probMat = (probMat.T / sumRel).T
            probMat = probMat*np.array(pis)
            
            params = {'cov':cov, 'means': means, 'pis' : pis, 
                          'probMat':probMat, 'Gmms': Gmms}
            
            return params
        
    def calcGmmPar(self, X, memb):
        #CALCULATES PARAMETERS FOR EACH GAUSSIAN
        #FOR EACH CLUSTER
        #RETURNS:
        #covk : covariance matrix of gaussian of class k
        #meank : mean vector of gaussian of class k
        #pk: mixing coefficient of gaussian of class k
        #model : the Gaussian of class k (object)
        #proba: the posterior probabilities, i.e probabilities of being
        #in class k given X 
        
        Nk = np.sum(memb)
        N = X.shape[0]
        
        pk = Nk/N  #mixing coefficient
       # print("Minimum of memb {} max{}".format(np.min( memb), np.max( memb)))
        meank = np.sum( ( X.T * memb ).T, axis = 0) / Nk
        covk =  memb*( X - meank ).T @ ( X - meank) + np.eye(X.shape[1])*10**(-4)
        covk = covk/Nk
       # eige = np.linalg.eigvals(covk)
       # print(np.min(eige))
        model  = multivariate_normal(meank, covk)
        logproba = model.logpdf(X) 
        
       # print("min log prob {}, max of log prob {}".format( np.min(proba ), np.max(proba )))
        #proba = model.cdf(X)*pk
        return covk, meank, pk, logproba, model
    
    
    def predict_proba(self, Xtest = None, Xtrain = None):  
        
        """
           AFTER FITTING THE MODEL IT PREDICTS THE PROBABILITY ON  THE TEST
           SET THE MODEL WAS FITTED 
           
        """
        #CHECKING IF THE MODEL IS FITTED 
        if self.mTest == None:
            print("The Model is not fitted or some other error might have\
                              occured")
            return
        
       
        logisticModels = self.LogRegr
        
        #PROBABILITY MATRIX THE METHOD WILL RETURN
        #SPECIFICALLY EACH ENTRANCE WILL HAVE THE PROBABILITY 
        #EACH DATAPOINT TO BE 1
        pMatrixTest = np.zeros( (Xtest.shape[0]) )
        pMatrixTrain = np.zeros( (Xtrain.shape[0]) )
        
        #FOR EACH MODEL CALCULATE THE PREDICTION FOR EACH DATA POINT
        for i, model in enumerate( logisticModels ):
            probsTest = model.predict_proba(Xtest)[:,1] #probability each point
                                                    #to be in class 1
                                                    
            probsTrain = model.predict_proba(Xtrain)[:,1] #probability each point
                                                    #to be in class 1
            pMatrixTest += probsTest*self.mTest[:, i]
            pMatrixTrain += probsTrain*self.mTrain[:, i]
            
            
        return pMatrixTest, pMatrixTrain
            
                
        
        
        
        
        
        
        
        
        
            