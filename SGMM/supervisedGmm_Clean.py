#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:14:37 2019

@author: george
"""

import sys 

sys.path.append('..')
#sys.path.append('../SGMM')
sys.path.append('../metrics')
sys.path.append('../loaders')
sys.path.append('../oldCode')
sys.path.append('../visual')
sys.path.append('../testingCodes')
sys.path.append('../otherModels')


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  SGDClassifier
from sklearn.model_selection import GridSearchCV
from  scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from cvxopt import matrix
from cvxopt.solvers import qp
from sklearn.linear_model import LogisticRegression

####THIS CODE HAS NUMERICAL ISSUES AT THE SPARCS DATASET

class SupervisedGMM():
    
    """ 
        THIS CLASS IMPLEMENTS THE SUPERVISED GMM ALGORITHM 
        IT USES SOME PARTS OF SCIKIT LEARN TO ACCOMPLISH THIS    
    """
    
    
    def __init__(self, max_iter = 1000, cv = 5, mix = 0.2, 
                 C = [1/0.001, 1/0.01, 1/0.1, 1, 1/10, 1/100, 1/1000, 1/10000], 
                 alpha = [ 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000 ],
                 max_iter2 = 10, penalty = 'l1', scoring = 'neg_log_loss',
                 solver = 'saga', n_clusters = 2, tol = 10**(-3 ) , 
                 mcov = 'diag', tol2 = 10**(-3), transduction = 0, adaR = 0,
                 verbose = 1, warm = 0,  altern = 0, log_reg = 'LG'):
        
        
        
        """ MODEL  PARAMETERES:
            
            max_iter:[INT] #Number of epochs of SGD default 1000
            
            cv:[INT] Cross Validation: Default 5 Fold
            
            mix:{FLOAT][0-1] In what Percentages to Upadate Memberships in respect with
            the previous iteration: Def --> 0.5
            
            
            C: [LIST] Inverse of Regularization Parameter: DEF: 1000
           
            alpha:[LIST] regularization parameters, for the stochastic 
            gradient descend cross validation
            
            
            max_iter2: Maximum # of EM Iterations, DEF: 10 
            
            penalty:[STRING] Regularization type ( Default L1 )
            
            scoring:[STRING] score to optimize in cross validation: DEF: 
                'negative_log_loss
                
            solver: [STRING] DEF: 'saga', Solvers used by scikit learn
            for logistic Regression 
            
            n_clusters:{INTEGER] #of Soft Clusters: DEF: 2
    
            tol:[FLOAT] memberships convergence tolerance
            
            tol2 =[FLOAT] stochastic gradient descent tolerance: Def 10^(-3)
            
            mcov =[STRING] 'full' or 'diag', 'full' means full covariance,
                   'diag' means diagonal covariance
                   
            transduction:[BINARY] 
            If to use transductional logic or not: Default: 1(TR)
            
            adaR:[BINARY] Adaptive regularization , regularize according to the
            cluster size
            
            verbose:[BINARY] information on the fitting of the algorithm and 
            other information
            
            warm:[BINARY], if this is 1 we need to give during the fitting the
            memberships, warm start, given memberships
            
            altern: [BINARY] start using  prediction driven approach when
                            memberships have converged with just mixture models
            
            log_reg: [STRING], "SGD" stochastic gradient descend,
                               "LG" Logistic Regression
            
                
    
        """
        ######################################################################
        # CLASS ATTRIBUTES
        ######################################################################
        #ind1 and ind2 are optional feature selection parameters that might
        #be specified in the fit method
        self._ind1 = None
        self._ind2 = None
        #idx1, idx2 are indexes created if we do a split and explained in
        #the split method
        self._idx1 = None 
        self._idx2 = None
        
        #maximum number of epochs for SGD
        self._max_iter = max_iter
        #Number of Folds for Cross Validation
        self._cv = cv
        #Mixing Coefficient
        self._mix = mix
        #NOT USED ANY MORE
        self._C = C
        #List with regularization parameters for cross validation
        self._alpha = alpha
        #Nuber of iterations of the EM algorithm
        self._max_iter2 =  max_iter2
        #type of penalty for logistic regression
        self._penalty = penalty
        #type of scoring  for cross validation
        self._scoring = scoring
        #type of scikit learn solver for SGD
        self._solver = solver
        #nimber of clusters to use 
        self._n_clusters = n_clusters
        #tolerance for the SGD agorithm
        self._tol = tol
        #tolerance for the membership convergence
        self._tol2 = tol2
        #variable for the type pf covariance for the gaussians
        self._mcov = mcov
        #use transuction or not
        self._trans = transduction
        #use adaptive regularization of not
        self._adaR = adaR
        #verbose or not
        self._vb = verbose
        #warm : warm start or not
        self._warm = warm
        

        #binary defaults to 0 (in general use 0)
        self._altern = altern
        #type of logistic regression algorithm to use
        self._log_reg = log_reg
        
        #FOR FIT INITIALIZE WITH KMEANS THE MEMBERSHIPS
        self._KMeans = None
        
        ######################################################################
        
        #THE FOLLOWING ATTRIBUTES ARE SETTED AFTER FITTING THE ALGORITHM
        #TO DATA
        
        
       
        #A list of the Gaussians After fitting the data
        #they can be used to predict memebership of a new data points
        self.Gmms = None   
        #mixture coefficients of Gaussians 
        self.mixes = None  
        #A list of Logistic regression predictors one for each class
        self.LogRegr = None 
        
        #PARAMETERS OF THE GAUSSIANS MODELS
        #list of covariances matrixes, list of means, list of mixes
        #probability matrix "Gauss membershipfor train and test if test iused
        #Gmms list of Gaussian predictors from Scipy Class
        self.params = None  
        
        #PARAMETERS AFTER FITTING THE MODEL
        #fitParams = {'mTrain' : mTrain, 'mTest': mTest, 'labTest': testlabels,
        #             'labTrain' : trainlabels }
        #memberships for training data and testing data
        #hard cluster labels for training and testing data
        self.fitParams = None 
                             
                             
       
        #GAUSSIAN'S MEANS AND COVARINACES
        self.means = None 
        self.cov = None
        #LOGISTIC REGRESSION WEIGHTS
        self.weights = None
        
        #TRAIN AND TEST MEMBERSHIPS SOFT
        self.mTrain = None
        self.mTest = None
        
        #IF MODEL IS FITTED OR NOT
        self.fitted = None
        
        #For CDPHP MOUAD AND MATT
        #A monitor attribute recording in each EM ITERATION the
        #memberships of the training points and the probability of each
        #point to be 1
        self.monitor_ = {'memb_3D': [], 'prob2D':[]}
        
        
        
    def fit(self, Xtrain = None, ytrain = None, Xtest = None, ind1 = None,
                    ind2 = None, mTrain1 = None, mTest1 = None, 
                    kmeans = 1, mod = 1, simple = 0, comp_Lik = 0,
                    memb_mix = 0.0, hard_cluster = 0):
        """ 
            Fit the Supervised Mixtures of Gaussian Model
            
            ind1: chose the features to use in the training of the Ml model
            
            ind2: chose the fetures to use in the training of the Gaussians
            
            Xtrain: training data
            
            ytrain: labels of training data
            
            Xtest: testing data if tranduction is on
            
            kmeans: kmeans initialization  of memberships
            
            mod: mode of computing the probabilities for gaussians, default
                 mod = 1
            
            simple : binary variable to decide if you will use simple 
            mixture of gaussians plus classification [simple = 1], if 
            simple is 0 [simple = 0] then use prediction driven gaussians
            for training ( the proposed model )
            a third choice is to  use simple  = 0  and set the altern variable 
            from the model to 1 this will use no prediction driven results 
            till some convergence of the gaussian and then 
            it will alter to prediction driven model
            
            comp_Lik: (UNSTABLE) Compute Likelihood  or not 
            
            memb_mix: parameter of mixing between predicted and learned 
                      memberships
            
            hard_cluster: hard cluster memberships before the logistic regression
                          fitting.
            
            
        """
        #CHECK IF ALL DATA ARE GIVEN
        self.ind1 = ind1
        self.ind2 = ind2
        self.fitted = 1
        self._KMeans = kmeans
        
        if Xtrain is None or ytrain is None  :
            print(" Please Give Xtrain, ytrain, Xtest  data ")
            return
        
        if ind1 is None:
            #ALL FEATURES FOR PREDICTION IF ind1 NOT SPECIFIED
            self.ind1 = np.arange( Xtrain.shape[1] )
            ind1 = self.ind1
            
        if ind2 is None:
            #ALL FEATURES FOR CLUSTERING IF ind2 NOOT SPECIFIED
            self.ind2 = np.arange( Xtrain.shape[1] )
            ind2 = self.ind2
        
        #PARAMETERES TO BE USED BY THE FIT FUNCTION
        n_clusters = self._n_clusters
        max_iter = self._max_iter
        cv = self._cv
        mix = self._mix
        penalty = self._penalty
        scoring = self._scoring
        #solver = self._solver
        max_iter2 = self._max_iter2
        trans = self._trans
        C = self._C
        alpha = self._alpha
        tol = self._tol
        tol2 = self._tol2
        mcov = self._mcov
        adaR = self._adaR
        vb = self._vb
        warm = self._warm
        altern = self._altern
        lg_regr = self._log_reg
        
        dimXtrain = Xtrain.shape[0]
        dimXtest = 0
        
        if trans == 1:
            dimXtest = Xtest.shape[0]
            
        #regularize the sums  for numerical instabilities
        reg = 10**(-3)
        #regularization to be added to every memebership entry
        regk = reg/n_clusters
        
        
        
        
        #INITIALIZE MEMBERSHIP FUNCTIONS
        #WE KEEP SEPARATED TRAIN AND TEST MEMBERSHIPS
        #BECAUSE TRAIN IS SUPERVISED MEMBERSHIP
        #TEST IS UNSUPERVISED
        
        mTrain, mTest = self.initializeMemb(warm, kmeans, dimXtrain, n_clusters,
                       regk, trans, dimXtest, Xtrain, Xtest, mTrain1,
                       mTest1)


        #NORMALIZE MEMBERSHIPS SO EACH ROW SUMS TO 1
        sumTrain = np.sum( mTrain, axis = 1) 
        mTrain = ( mTrain.T / sumTrain ).T
        
        if trans == 1:
            sumTest = np.sum( mTest, axis = 1 )
            mTest = ( mTest.T / sumTest ).T    
        
       
        
        #SET SOME  PARAMETERES FOR USE IN THE FOR LOOP OF EM
        indexing = np.arange( dimXtrain )
        
        #MATRIX WITH LOGISTIC REGRESSION PROBABILITIES FOR EACH CLASS
        logiProb = np.zeros([dimXtrain, n_clusters])
        
        #MATRIX WITH LOG PROBABILITIES
        logLogist = np.zeros([dimXtrain, n_clusters])
       
        #variable related  to the altern parametere
        gate = 0
        ###START FITTING ALGORITHM ##################
       
        #setting the cross validation grid
        if lg_regr == 'SGD':
            param_grid = {'alpha': alpha}
        else:
            param_grid = {'C':  C}
            
        Qold = 0 #initial likelihood (if we are to calculate it)
        
        #STARTING EM ALGORITHM
        for iter2 in np.arange( max_iter2 ): #OUTER EM ITERATIONS
            #FITING THE L1 LOGISTIC REGRESSIONS
            if simple == 0:
                models, logiProb, logLogist = self.fitLogisticRegression( 
                              n_clusters, mTrain, adaR, alpha, max_iter,
                              tol2, Xtrain, ytrain, vb, penalty, scoring,
                              cv, regk, ind1, indexing, logiProb, logLogist, 
                              param_grid,  lg_regr, C,
                                              hard_cluster = hard_cluster )
                
            else: #IF WE USE SIMPLE MIXTURES OF GAUSSIANS JUST FIT AT LAST ITER
                if iter2 == ( max_iter2 - 1):
                    models, logiProb, logLogist = self.fitLogisticRegression( 
                              n_clusters, mTrain, adaR, alpha, max_iter,
                              tol2, Xtrain, ytrain, vb, penalty, scoring,
                              cv, regk, ind1, indexing, logiProb, logLogist, 
                              param_grid, lg_regr, C,  hard_cluster = hard_cluster )
            
            
            #set the models attribute after fitting the Logistic Regression
            #for all clusters
            self.LogRegr = models
            
            #WE TAKE THE MEMBERSHIPS AND ALL THE DATA
            #TO FIT THE GAUSSIANS USING THE EM ALGORITHM FOR GMM 
            if trans == 1: #if we have transduction
                data = np.concatenate((Xtrain[:, ind2], Xtest[:, ind2]), 
                                                                      axis = 0)
                mAll = np.concatenate( (mTrain, mTest ), axis = 0 )
            
            else:
                 data =  Xtrain[:, ind2]
                 mAll = mTrain
               
            #take the parameters of the GMM models
            #THIS PIECE OF CODE WILL BE REMOVED IN THE FUTURE
            params = self.gmmModels( data, mAll, mcov )
            gmmProb = params['probMat']
            ###################################################################
            #THIS IS AFTER MODIFICATIONS#######################################
            if mod == 1: #THIS IS THE MOD WE WILL KEEP IN THE FUTURE
                gmmProb = params['probMat2']
                gmmLogprob = params['logProb']  
                self.Gmms = params['Gmms']
                self.mixes = params['pis']
            #END OF MODIFICATION ##############################################
            
           #CALCULATE NEW MEMBERSHIPS FOR TRAIN AND TEST
            if simple and gate == 0: #NO PREDICTION DRIVEN (MoG + LR + L1)
                mNewTrain =  gmmProb[0: dimXtrain, :] + regk
                
            else: #PREDICTION DRIVEN (SGMM)
                mNewTrain = logiProb * gmmProb[0: dimXtrain, :] + regk
                
                simple = 0
            
            ###################################################################
            if trans:
                mNewTest = gmmProb[dimXtrain :, :] + regk
             
            #COMPUTE LIKELIHOOD IF COMP_LIK == 1
            if mod  and comp_Lik:
                 Qnew, Qdif = self.computeLikelihood( gmmLogprob, logLogist,
                                    dimXtrain, vb, trans, simple, Qold)
                 Qold = Qnew
            #END OF MODIFICATION ##############################################
            
            #NORMALIZE NEWMEMBERSHIPS
            sumTrain = np.sum( mNewTrain, axis = 1)
            if trans == 1:
                sumTest = np.sum( mNewTest, axis = 1 )
          
            mNewTrain = ( mNewTrain.T / sumTrain ).T
            if trans == 1:
                mNewTest = ( mNewTest.T / sumTest ).T  
                
                
            #EVALUATE ERROR
            errorTr = np.sum( np.abs( mTrain - mNewTrain) )
            if trans == 1:
                errorTst = np.sum( np.abs( mTest - mNewTest ) )
                error = ( errorTr + errorTst )/( (dimXtrain + dimXtest)\
                                                             *n_clusters )
            else:
                error = errorTr/( dimXtrain * n_clusters )
            
            if (error < 5*10**(-8)) and altern:
                gate = 1
                altern = 0
                
            #MAKE A SOFT CHANGE IN MEMEBRSHIPS MIXING OLD WITH NEW 
            #MEMBERSHIPS WITH DEFAULT MIXING OF 0.5
            mNewTrain = mNewTrain*(1-memb_mix) + \
                                            self.predict_GMMS(Xtrain)*memb_mix
                                            
            mTrain = mNewTrain*(1-mix) + mTrain*(mix)
            if trans == 1:
                mTest = mNewTest*(1-mix) + mTest*(mix)
        
            ###### CDPHP MOUAD AND MATT ----> MONITOR IMPLEMENTATION #########
            
            
            self.monitor( Xtrain, ytrain,  iter2 )
            
            
            ##################################################################
            print("GMM iteration: {}, error: {}".format(iter2, error))
            if error < tol:
                
                 break
        ############ END OF EM UPDATES #######################################
        #if simple  and error < tol:
        models, logiProb, logLogist = self.fitLogisticRegression( 
                              n_clusters, mTrain, adaR, alpha, max_iter,
                              tol2, Xtrain, ytrain, vb, penalty, scoring,
                              cv, regk, ind1, indexing, logiProb, logLogist, 
                              param_grid, lg_regr, C,
                                                  hard_cluster = hard_cluster )
        
        self.Gmms = params['Gmms']
        self.mixes = params['pis']
        self.LogRegr = models
        self.params = params        
        #TAKING HARD CLUSTERS IN CASE WE WANT TO USE LATER  
        if trans == 1:
            testlabels = np.argmax( mTest, axis = 1 )
            
        else:
            testlabels = []
            
        trainlabels = np.argmax( mTrain, axis = 1 )
        fitParams = {'mTrain' : mTrain, 'mTest': mTest, 'labTest': testlabels,
                     'labTrain' : trainlabels }
        
        self.mTrain = mTrain
        
        if trans == 1:
            self.mTest = mTest
            
        self.fitParams = fitParams
        #set the weights of LOGREG MEANS AND COVARIANCES OF GAUSSIANS
        self.setWeights()
        self.setGauss( params )
        
        return self
        #END OF FIT FUNCTION##################################################
        
    def initializeMemb( self, warm, kmeans, dimXtrain, n_clusters,
                       regk, trans, dimXtest, Xtrain, Xtest, mTrain1,
                       mTest1):
        
        """ Function to initialize memberships,
        warm: if we want a warm start ((provide mTrain1, mTest1))
        kmeans: [binary] kmeans initialization or not
        dimXtrain: number of training data
        n_clusters: number of clusters we use
        regk: amount of regularization for the divisions
        trans: use transduction or not (if yes we need test data too )
        if we have trunsduction give the dimension of test data
        Xtrain: training data
        Xtest: testing data
        mTrain1: given thatwe want a warm start give the initial memeberhsips
        mTest1: given that we want a warm start give the initial memeberships
        of test data
        """
        
        
        
        if warm == 0: #IF WE DONT HAVE WARM START
            
            if kmeans == 0: #NO KMEANS INITIALIZATION (RANDOM INIT)
               mTrain = np.random.rand( dimXtrain, n_clusters) + regk
               
               if trans == 1:
                   mTest = np.random.rand( dimXtest, n_clusters )  + regk
                   
               else:
                   mTest = []
            
            else: #KMEANS INITIALIZATION
                km = KMeans( n_clusters = n_clusters, random_state = 0)
                if trans == 1:
                    #FIT KMEANS IN DATA (TEST AND TRAIN IF TRANSDUCTION)
                    km = km.fit( np.concatenate( (Xtrain, Xtest), axis = 0))
                    mAll = np.zeros([ dimXtrain +dimXtest, n_clusters])
                    
                else:
                    #FIT ONLY TRAIN IF NOT TRANSDUCTION
                    km.fit( Xtrain)
                    mAll = np.zeros([ dimXtrain , n_clusters])
                
                #TAKE THE LABELS FROM KMEANS
                labels = km.labels_
                for j in np.arange( labels.shape[0] ): #MAKE THE MEMBERSHIPS
                    mAll[j, labels[j]] = 1
                    
                mTrain = mAll[0: dimXtrain ]
                if trans == 1:
                    mTest = mAll[ dimXtrain :]
                    
                else:
                    mTest = []

        else: #IF WE HAVE WARM START, ASSIGN WITH THE GIVEN MEMBERSHIPS

            mTrain = mTrain1
            mTest = mTest1
        
        return mTrain, mTest
    
    
################### FITTING LOGISTIC REGRESSION MODEL #########################     
    
    def fitLogisticRegression(self, n_clusters, mTrain, adaR, alpha, max_iter,
                              tol2, Xtrain, ytrain, vb, penalty, scoring,
                              cv, regk, ind1, indexing, logiProb, logLogist,
                              param_grid, lg_regr, C,  hard_cluster):
        
        """ FIT LOGISTIC REGRESSION FOR EACH CLUSTER 
            n_clusters: number of gaussians -- clusters
            mTrain: train data membership,
            adaR: to use or not adaptive regularization
            alpha: regularization parameteres list
            max_iter : number of epochs to train the stochastic gradient
            descend algorithm
            tol2: tolerance of SGD training
            Xtrain: training data
            ytrain: training labels
            vb: to print some info at eout related to adaptive regularization
            such us cluster size, new alphas etc
            penalty: penalty to use for training , default L1 norm
            scoring: scoring to use for training , Default neg log loss
            cv: number of folds for cross validation
            regk: regularization when computing log probabilities
            ind1: indexes to use for training (feature columns)
            indexing: a list with the indexes of the training data
            logiProb: an initialized matrix to put the logistic regression
            probabilities
            logLogist: an initialized matrix to put the log probabilities
            lg_regr: Choice of SGD or FULL Logistic Regression
            C: regularization for logistic regression
            hard_cluster: hard_cluster memebrships before the fit of
                        logistic regressions
            
            returns: models-> logistic regresion models
                     logiProb--> probabilities of a data point to belong in 
                     in its class given the cluster
                     logLogist--> the same as above but log probabilities
                     
           
            """
            
        mTrain = self.hardCluster( mTrain.copy(), hard_cluster)
        models = []
        for clust in np.arange( n_clusters ): #FITLOG REGR
                #FIT THE L1 LOGISTIC REGRESSION MODEL
                #CROSS VALIDATION MAXIMIZING BE DEFAULT THE NEGATIVE LOG LIKEHOOD
                
                #ADAPTIVE REGULARIZATION
                Nclus = np.sum( mTrain[:, clust], axis = 0 )
                if adaR == 1:
                    if lg_regr == 'SGD':
                        alphanew = (np.array( alpha ) / Nclus).tolist()
                        param_grid = {'alpha': alphanew}
                    else:
                        
                        Cnew = (np.array( C ) / Nclus ).tolist()
                        param_grid = {'C': Cnew}
                # PRINT SOME INFORMATION  
                if vb == 1:
                    #print Cluster Size
                    print('\n Cluster {} has Size {} of {}'.format( clust,
                          Nclus, mTrain.shape[0]))
                    if adaR == 1:
                        if lg_regr == 'SGD':
                            print('alpha is {} alphaNew {}'.format(alpha, alphanew))
                        else:
                            print('C is {} CNew {}'.format(C, Cnew))
       
                #TRAIN LOGISTIC REGRESSION MODEL
                if lg_regr == 'SGD':
                    mf = SGDClassifier(loss = "log", penalty = penalty, 
                                      n_jobs = -1, max_iter = max_iter,
                                      random_state = 0, tol = tol2)
                else:
                
                    mf = LogisticRegression( penalty = penalty, tol = tol2,
                                             random_state = 0, 
                                        max_iter = max_iter, n_jobs = -1)
                 
                model = GridSearchCV( mf, param_grid = param_grid, 
                                  n_jobs = -1, 
                                  scoring = scoring, cv = cv).\
                                  fit(Xtrain, ytrain,
                                      sample_weight = mTrain[:, clust] ) 
    
                #FOR EACH CLUSTER APPEND THE MODEL in MODELS
                models.append( model )  
               
                #PREDICT PROBABILITIES FOR BEING IN CLASS 1 or 0
                proba = model.predict_proba( Xtrain[:, ind1] )
                #log probabilities
                logproba = np.log( proba + regk)
                
                #FOR EACH DATA POINT TAKE THE PROB ON BEING IN CORRECT CLASS
                logiProb[:, clust]  = proba[ indexing, ytrain ] 
                logLogist[:, clust] = logproba[ indexing, ytrain]
                
                ######## END OF CODE FITTING LOGISTIIC REGRESSION ############
        return models, logiProb, logLogist
   
    
################ COMPUTE LIKELIHOOD ##########################################    
    def computeLikelihood( self, gmmLogprob, logLogist, dimXtrain, vb, trans,
                                                                simple, Qold):
        
           """COMPUTER THE AUXILARY FUNCTION Q IN EACH ITERATION 
           gmmLogprob: The log probabilities for all clusters from Mixture
           of Gaussians
           logLogist:  Log probabilities from logistic regressin
           dimXtrain: Train Data Dimension
           vb:  verbose output,
           trans: if trunsduction is used or not
           simple: if we use the MoG or the SGMM
           Qold: the previous calculated Q value
           """
           dimXtest = gmmLogprob.shape[0] - dimXtrain
           if trans == 0:
               Qf = gmmLogprob + logLogist*(1-simple)
               Qf2 = np.log( np.sum(np.exp( Qf ), axis = 1) )
               Qf3 = np.sum( Qf2 )/dimXtrain
                      
           else:
               Qft = gmmLogprob[0: dimXtrain,:] + logLogist*(1-simple)
               Qf2 = np.log( np.sum(np.exp( Qft ), axis = 1) )
               Qf31 = np.sum( Qf2 )           
                    
               Qftest =  gmmLogprob[dimXtrain:, :] 
               Qftest2 =  np.log( np.sum(np.exp( Qftest ), axis = 1) )
               Qftest3 = np.sum( Qftest2 ) 
                    
               Qf3 = (Qftest3 + Qf31)/(dimXtest + dimXtrain)
           
           Qdif = abs( Qf3 - Qold)
           if vb == 1:     
               print("\n Qnew is : {}".format( Qf3 ))
              # print("\n Qdif is : {}".format( Qdif ))
           
           
           return Qf3, Qdif

################# #FITTING THE GAUSSIAN MIXTURE MODEL #########################            
    def gmmModels(self, X, members, mcov ):
            
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
            #GETTING THE NUMBER OF CLUSTERS " GAUSSIANS IN THE MIXTURE "    
            clusters = members.shape[1]
            regk = (10**(-5)/clusters)
            cov = [] #list with covariance matrices
            means = [] #list of means
            pis = [] #list of mixing coefficients
            
            #MATRIX WITH THE PROBABILITIES p(x_i/z_i = k) For all gaussians
            probMat = np.zeros( [X.shape[0], clusters] )
            #THE ACTUAL MODELS FOR PREDICTION OF THE MEMBERSHIPS ON TEST POINTS
            Gmms = []
            #MATRIX WITH THE LOG PROBABILITIES
            logprobaMatrix = np.zeros([X.shape[0], clusters])
                    
            for cl in np.arange( clusters ):
               
                # FOR EACH CLUSTER USE THE EM ALGORITHM
                # TO CREATE THE NEW MEMBERSHIP MATRIX OF THE GAUSSIANS
                #IT IS NOT EXACTLY THE MEMBERSHIP BECAUSE IT IS
                # NORMALIZED  AFTER THIS FUNCTION ENDS
                covCl, mCl, piCl, logproba, model = self.calcGmmPar( X, 
                                                                members[:,cl],
                                                                mcov) 
                
                #LOG PROBABILITIES FOR EACH CLUSTER
                logprobaMatrix[:,cl] = logproba 
                
                #APPEND GAUSSIAN STATS
                cov.append( covCl )
                means.append( mCl )
                pis.append( piCl )
                Gmms.append( model )
                
            #FOR EACH DATA POINT FIND THE MAXIMUM LOGPROBABILITY 
            #THIS IS DONE FOR REGULARIZATION PURPOSES
            maxLog = np.max( logprobaMatrix, axis = 1 )
            logprobaMatrix2 = ( logprobaMatrix.T - maxLog).T

            
            #### NEXT 4 LINES BEFORE  WILL BE DELETED IN FUTURE
            probMat = np.exp( logprobaMatrix2 ) + regk
            sumRel = np.sum( probMat, axis = 1)
            probMat = (probMat.T / sumRel).T
            probMat = probMat*np.array(pis)
 
            #THIS WILL BE KEPT IN THE FUTURE -->p(x/z_i)p(z_i)
            probMat2 = np.exp( logprobaMatrix2 )*np.array( pis ) + regk
            totLog = logprobaMatrix + np.log( np.array( pis ) )
            
            params = {'cov':cov, 'means': means, 'pis' : pis, 
                          'probMat':probMat, 'Gmms': Gmms, 'probMat2': probMat2,
                          'logProb': totLog}
            
            return params
        

    def calcGmmPar(self, X, memb, mcov):
        """CALCULATES PARAMETERS FOR EACH GAUSSIAN
        #FOR EACH CLUSTER
        #RETURNS:
        #covk : covariance matrix of gaussian of class k
        #meank : mean vector of gaussian of class k
        #pk: mixing coefficient of gaussian of class k
        #model : the Gaussian of class k (object)
        #proba: the posterior probabilities, i.e probabilities of being
        #in class k given X 
        """
        
        
        
        reg = 10**(-3)         #regularization for Covariances
        Nk = np.sum(memb)      #Cluster Population
        N = X.shape[0]         #Number of data Points
        
        #mixing coefficient
        pk = Nk/N  
        meank = self.cMean(X, memb, Nk)
        covk = self.cCov(X, meank, memb, reg, Nk, mcov)
        
        
                
        model  = multivariate_normal( meank.copy(), covk.copy() )
        logproba = model.logpdf(X)  #LOG PROBABILITY cansuisthebestofthebest
        
        return covk, meank, pk, logproba, model
    
    
           
    #----------------FUNCTIONS TO PREDICT MEMBERSHIPS AND PROBABILITIES -------
    def predict_proba(self, X = None):
        """
        Predicts the Probabity of  data--> X to be in class 1
        X input array
        
        Returns the probability of each point on X to belong in class 1
        
        p(y = 1| x, M)
        
        """
        
        #get the Logistic Regression Predictors
        models = self.LogRegr
        #predict the membership matrix of X
        memb = self.predict_GMMS( X )    
        #take the total probability 
        totalProb = np.zeros( [X.shape[0]])
        
        for i in np.arange( memb.shape[1] ):
            #probability  points of X belong in class 1
            model = models[i]
            probModel = model.predict_proba( X )
            proba = probModel[:, 1]
            totalProb += proba*memb[:, i]
        
        return totalProb 
        
    
    def predict_GMMS( self, X):
        """
        Given a Data matrix X it returns the Membership matrix 
        for each data point in X based on the Gaussians already fitted
        
        """
        
        if self.fitted == 0:
            print("Warning: There is no fitted model ")
            return []
        
        gmms = self.Gmms
        mixes = self.mixes
        regk = 10**(-5)/len( gmms )
        
        membership = np.zeros( [X.shape[0], len( gmms )] )
        logmembership = np.zeros( [X.shape[0], len( gmms )] )
        for i in np.arange( len( gmms ) ):
            
            logmembership[:, i] =  gmms[i].logpdf( X[:, self.ind2] )#*mixes[i]
            
        maxlog = np.max( logmembership, axis = 1)
        logmembership = (logmembership.T - maxlog).T
        probMat = np.exp( logmembership )* np.array( mixes ) + regk
        sumRel = np.sum( probMat, axis = 1)
        membership = (probMat.T / sumRel).T 
       
        return membership
     
    #HELPER FUNCTIONS USED IN THE FIT METHOD    
    def getweightsL1(self, models ):
        """GIVEN THE LOGISTIC REGRESSION MODELS,
        RETURN THE SUM OF THE WEIGHTS PLUS REGULARIZATION """
        sumW = 0
        
        for i, model in enumerate( models ):
            weights = model.best_estimator_.coef_.tolist()[0]
            alphDict = model.best_params_
            alph = alphDict['alpha']
            weights = np.array( weights )
            weights = np.abs( weights )
            sumW += alph*np.sum(weights)
            
        return -sumW
         
    def setWeights( self ):
        """ setting logistic regression weights for each cluster """
        if self.fitted == None:
            print("MODEL IS NOT FITTED YET")
            
        models = self.LogRegr
        
        weights = []
        for model in models:
            weight = model.best_estimator_.coef_.tolist()[0]
            intercept = model.best_estimator_.intercept_[0]
            weight.insert(0, intercept)
            weights.append( weight )
        
        self.weights = weights
        
        return
    
    
    def setGauss( self, params ):
        #SETTING MEANS AND COVARIANCES OF THE GAUSSIANS
        if self.fitted == None:
            print("MODEL IS NOT FITTED YET")
            
        self.means = params['means']
        self.cov = params['cov']
        self.pis = params['pis']
        return
        
    def hardCluster( self, mTrain, hard_cluster):
        """takes the memeberships assigns 1 at the max element of each row
         and 0 to all the other elements of the row
        """
        if hard_cluster:
            mTrain2 = np.zeros_like( mTrain )
            mTrain2[ np.arange(len(mTrain)), np.argmax( mTrain, axis = 1)] = 1
            return mTrain2
        
        return mTrain
    
    
    ########## HELPER FUNCTIONS FOR MEANS AND COVARIANCES #########################  
    def cMean(self, X, memb, Nk):
        
        """calculates the weighted mean for gaussian k"""
        
        meank = np.sum( ( X.T * memb ).T, axis = 0) / Nk
        return meank
    
    def cCov(self, X, meank, memb, reg, Nk, mcov):
        
        """Given a data Matrix X, its weighted mean, the membership
        vector, a regularization parameter,  the type of covariance, full or diagonal
        and the weighted sample size
        calculates the weighted covariance matrix for gaussian k,
       
        """
       
        if mcov == 'full':
            covk = (memb*( X - meank ).T)@ ( X - meank) \
                                                + np.eye(X.shape[1])*reg
        else:#diagonal covariance
            covk = np.sum( memb*( np.square( X-meank ).T ), axis = 1 ) 
            covk = np.diag( covk )  + np.eye(X.shape[1])*reg
        
        covk = covk/Nk
        
        return covk
            
    ####### END OF HELPER FUNCTIONS FOR MEANS AND COVARIANCES #################     
    def monitor( self, X, Y, i):
        """
        
        Calculates the memberships per EM iteration of the matrix X
        and puts them in a 3D 
        matrix  MT : shape: (N,K,T
        
        Calculates the probability of every entry on matrix X to be 1
        and puts them in an
        matrix : PT:(N,2,T) 
        where the Nx2 matrix will have the probabilities 
        and the corresponding labels
        
        N: Number of Points in X
        K: Number of Gaussian Mixtures
        T: Number of EM_iterations
        
        Sets the attribute
        self.monitor_ 
        with the aformentioned matrices in the form of a dictionary
        with keys:  'memb_3D', 'prob2D'
            ex:
            self.monitor_ = {'memb_3D': [], 'prob2D':[]}
            
            
        """
        N = X.shape[0]
        K = self._n_clusters
        T = self._max_iter2
        
        if i == 0:
            MT = np.zeros( shape = [N,K,T])
            PT = np.zeros( shape = [N,2,T])
        
        else:
            MT = self.monitor_['memb_3D']
            PT = self.monitor_['prob2D']
        
        
        memb = self.predict_GMMS(X)
        proba = self.predict_proba(X = X)
        
        MT[:,:,i] = memb
        PT[:, 0, i] = proba
        PT[:, 1, i] = Y
        
        self.monitor_['memb_3D'] = memb
        self.monitor_['prob2D'] = PT
        
        return self
        
        
        
       
           
           
    







        
            
