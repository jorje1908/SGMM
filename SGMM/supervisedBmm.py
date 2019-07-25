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
#----------------------TO DO ----------------------------------------------
#--------------------CLEAN THE CODE REMOVE THE SGMM -----------------------
#-------------------- make the code  --------------------------------------

class SupervisedBMM():
    
    """ 
        THIS CLASS IMPLEMENTS THE SUPERVISED BMM ALGORITHM 
        IT USES SOME PARTS OF SCIKIT LEARN TO ACCOMPLISH THIS    
    """
    
    
    def __init__(self, max_iter = 1000, cv = 5, mix = 0.2, 
                 C = [1/0.001, 1/0.01, 1/0.1, 1, 1/10, 1/1000, 1/10000], 
                 alpha = [ 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000 ],
                 max_iter2 = 10, penalty = 'l1', scoring = 'neg_log_loss',
                 solver = 'saga', n_clusters = 2, tol = 10**(-3 ) , 
                 mcov = 'diag', tol2 = 10**(-3), transduction = 0, adaR = 0,
                 verbose = 1, warm = 0, m_sparse = 0, m_sparseL = 10**(-3),
                 m_sp_it1 = 2, m_sp_it2 = 2, m_choice = 0, 
                 m_LR = 0.001, m_mix = 1, altern = 0, log_reg = 'LG'):
        
        
        
        """ MODEL  PARAMETERES:
            
            max_iter:[INT] #Number of epochs of SGD default 1000
            cv:[INT] Cross Validation: Default 5 Fold
            mix:{FLOAT] In what Percentages to Upadate Memberships in respect with
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
            m_sparse:{BINARY] take sparse means in the gaussians centers or not
            m_sparseL: [FLOAT] the lambda for the means regularization
            m_sp_it1: iterations to run the first sparse means algorith if 
            chosen
            m_sp_it2: iterations to run the second sparse algorithm if chosen
            m_choice:[BINARY] choice of sparse algorithm QP or Gardient Descend
            m_LR:  if the choice if Gradient descend  pick the learning rate
            m_mix: 
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
        
        self._m_sparse = m_sparse
        self._m_sparseL = m_sparseL
        self._m_sp_it1 = m_sp_it1
        self._m_sp_it2 = m_sp_it2
        self._m_choice = m_choice 
        self._m_LR =  m_LR
        self._m_mix = m_mix
        self._altern = altern
        self._log_reg = log_reg
        
        #FOR FIT INITIALIZE WITH KMEANS THE MEMBERSHIPS
        self._KMeans = None
        
        ######################################################################
        
        #THE FOLLOWING ATTRIBUTES ARE SETTED AFTER FITTING THE ALGORITHM
        #TO DATA
        
        #PARAMETER TO BE SETTED AFTER THE MODELS IS FITTED
       
        #A list of the Gaussians After fitting the data
        #they can be used to predict memebership of a new data points
        self.Gmms = None   
         
        #when we fit a mixture of bernulis the means of the bernullis           
        self.Bers = None 
        #mixture coefficients of Gaussians or/and Bernullis (to do )
        self.mixes = None  
        self.mixesB = None
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
        
        ######################################################################
        #TO DO 
        # INCORPORATE BERNULLIS AND GAUSSIANS TOGETHER IN THE SAME FIT 
        # FUNCTION
        #GIVE THE BINARY DATA COLUMNS AND USE THESE FOR BERNULLIS AND THE
        #REST WITH GAUSSIANS
        #######################################################################
    #HELPER   
    def split(self, data = None, X = None, y = None, split = 0.2):
        """
        A helper function to split data into training and test set
        There are 2 choices, either Input a data numpy Array with the last 
        column be its labels or  The data and the labels separately
        data: Data with last column labels
        X: Data
        y: labels
        split: the percentage of test data
        
        returns: Xtrain, Xtest, ytrain, ytest, idx1, idx2
        idx1:  indexes taken for training data
        idx2:  indexes taken for test data
        
        
        """
        
       # if (data and X and y ) is None:
          #  return "Input data or X and y "
        
        if (X is None) and (y is None):
            Xtrain, Xtest, ytrain, ytest , idx1, idx2  = \
                                train_test_split(data[:,:-1], data[:,-1], 
                                 np.arange( data.shape[0] ), 
                                 test_size = split, random_state = 1512,
                                 stratify = y)
        else:
            Xtrain, Xtest, ytrain, ytest, idx1, idx2 = \
                                train_test_split(X, y, 
                                 np.arange( X.shape[0] ), 
                                 test_size = split, random_state = 1512,
                                 stratify = y)
        self.idx1 = idx1
        self.idx2 = idx2
        
        return Xtrain, Xtest, ytrain.astype(int), ytest.astype(int)
        
    
          
        
    def hardCluster( self, mTrain, hard_cluster):
        """takes the memeberships assigns 1 at the max element of each row
         and 0 to all the other elements of the row
        """
        if hard_cluster:
            mTrain2 = np.zeros_like( mTrain )
            mTrain2[ np.arange(len(mTrain)), np.argmax( mTrain, axis = 1)] = 1
            return mTrain2
        
        return mTrain
            

##fitB UNDER CONSTRUCTION WORKING WITH BINARY DATA NOT SUFFICIENTLY TESTED              
    def fitB(self, Xtrain = None, ytrain = None, Xtest = None, ind1 = None,
                                                              ind2 = None):
        """ 
            Fit the Supervised Mixtures of Bernullies
            ind1: chose the features to use in the training of the Ml model
            ind2: chose the fetures to use in the training of the Bernoulis
            the same as fit but fitting bernoullis at binary features
            instead of gaussians
        """
        #CHECK IF ALL DATA ARE GIVEN
        self.ind1 = ind1
        self.ind2 = ind2
        
        if Xtrain is None or ytrain is None or Xtest is None :
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
       # solver = self._solver
        max_iter2 = self._max_iter2
        dimXtrain = Xtrain.shape[0]
        dimXtest = Xtest.shape[0]
       # Cs = self._Cs
        alpha = self._alpha
        tol = self._tol
        tol2 = self._tol2
       # mcov = self._mcov
        #regularize the sums  for numerical instabilities
        reg = 10**(-5)
        #regularization to be added to every memebership entry
        regk = reg/n_clusters
        
        
        
        
        #INITIALIZE MEMBERSHIP FUNCTIONS
        #WE KEEP SEPARATED TRAIN AND TEST MEMBERSHIPS
        #BECAUSE TRAIN IS SUPERVISED MEMBERSHIP
        #TEST IS UNSUPERVISED
        mTrain = np.random.rand( dimXtrain, n_clusters) + regk
        mTest = np.random.rand( dimXtest, n_clusters )  + regk


        #NORMALIZE MEMBERSHIPS SO EACH ROW SUMS TO 1
        sumTrain = np.sum( mTrain, axis = 1) 
        sumTest = np.sum( mTest, axis = 1 )
        mTrain = ( mTrain.T / sumTrain ).T
        mTest = ( mTest.T / sumTest ).T    
        
        #print(np.sum(mTrain, axis = 1), np.sum(mTest, axis = 1))
        
        #SET SOME  PARAMETERES
        #FOR USE IN THE FOR LOOP
        indexing = np.arange( dimXtrain )
        logiProb = np.zeros([dimXtrain, n_clusters])
       
       
        
        #START FITTING ALGORITHM
        #CORE
        #setting the cross validation grid
        param_grid = {'alpha': alpha}
        for iter2 in np.arange( max_iter2 ):
            #FITING THE L1 LOGISTIC REGRESSIONS
            models = [] #EVERY IETARTION CHANGE MODELS 
                        # IN ORDER TO TAKE THE MODELS OF LAST ITERATION
                        #OR THE MODELS BEFORE ERROR GO BELOW TOLERANCE
            for clust in np.arange( n_clusters ):
                                                                 
                #FIT THE L! LOGISTIC REGRESSION MODEL
                #CROSS VALIDATION MAXIMIZING BE DEFAULT THE F1 SCORE
                
                sgd = SGDClassifier(loss = "log", penalty = penalty, 
                                      n_jobs = -1, max_iter = max_iter,
                                      random_state = 0, tol = tol2)
                model = GridSearchCV( sgd, param_grid = param_grid, 
                                  n_jobs = -1, 
                                  scoring = scoring, cv = cv).\
                                  fit(Xtrain, ytrain) #fit model 
    
#                model = LogisticRegressionCV(Cs = Cs, penalty = penalty,
#                             scoring = scoring, random_state = 0, n_jobs = -1,
#                             solver = solver, max_iter =
#                             max_iter,cv = cv).fit( Xtrain[:, ind1], ytrain,
#                             mTrain[:, clust] )
                
                #FOR EACH CLUSTER APPEND THE MODEL in MODELS
                models.append( model )  
               
                #PREDICT PROBABILITIES FOR BEING IN CLASS 1 or 0
                #FOR THE TRAINING DATA
                proba = model.predict_proba( Xtrain[:, ind1] )
                
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
            data = np.concatenate( ( Xtrain[:, ind2], Xtest[:, ind2] ),axis = 0)
            mAll = np.concatenate( (mTrain, mTest ), axis = 0 )
            
            #params is  a dictionary with the following structure
            #params = {'cov':cov, 'means': means, 'pis' : pis, 
            #                 'probMat':probMat, 'Gmms': Gmms}
            #cov: list of covariances
            #means: list of means
            #pis : list of probabilities of a specific gaussian to be chosen
            #probMat: posterior probability membership matrix
            #Gmms ; a list of Object with the Gaussians for each class
            params = self.berModels( data, mAll )
                
            berProb = params['probMat']
            #SOME DEBUGGING
            #print(gmmProb)
           # if iter2 == 1:
             #return params
                
            #CALCULATE NEW MEMBERSHIPS FOR TRAIN AND TEST
            mNewTest = berProb[dimXtrain :, :] + regk
            mNewTrain = logiProb * berProb[0: dimXtrain, :] +regk
                
            #NORMALIZE NEWMEMBERSHIPS
            sumTrain = np.sum( mNewTrain, axis = 1)
            sumTest = np.sum( mNewTest, axis = 1 )
           # print(sumTrain, sumTest)
            mNewTrain = ( mNewTrain.T / sumTrain ).T
            mNewTest = ( mNewTest.T / sumTest ).T  
                
                
            #EVALUATE ERROR
            errorTr = np.sum( np.abs( mTrain - mNewTrain) )
            errorTst = np.sum( np.abs( mTest - mNewTest ) )
            error = ( errorTr + errorTst )/( (dimXtrain + dimXtest)*n_clusters )
            
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
            
                
            print("GMM iteration: {}, error: {}".format(iter2, error))
            if error < tol:
                 break
             
        
        self.mixesB = params['pis']
        self.LogRegr = models
        self.params = params        
        #TAKING HARD CLUSTERS IN CASE WE WANT TO USE LATER       
        testlabels = np.argmax( mTest, axis = 1 )
        trainlabels = np.argmax( mTrain, axis = 1 )
        
        fitParams = {'mTrain' : mTrain, 'mTest': mTest, 'labTest': testlabels,
                     'labTrain' : trainlabels }
        self.mTrain = mTrain
        self.mTest = mTest
        self.fitParams = fitParams
        
        return self        
        
        
##berModels UNDER CONSTRUCTION WORKING WITH BINARY DATA NOT SUFFICIENT TESTED        
    def berModels(self, X, members):
            
            """
                Calculates the Mixtures of Bernullis Parameters
                
                
                X : Train and Test data together
                members: Posterior Probabibilities for each cluster
                             and each data point (memberships)
                             
                Returns: a list with the mean matrices of the Bernullis,
                a list with the mixing parameteres,
                
                the probability matrix with the posteriors for each data
                point and each cluster,
                
                All these it returns in the form of a dictionary
                
            """
                
            clusters = members.shape[1]
            regk = (10**(-5)/clusters)
           
            means = [] #list of means Probabilities of clusteres
            pis = [] #list of mixing coefficients
            probMat = np.zeros( [X.shape[0], clusters] )
           
            logprobaMatrix = np.zeros([X.shape[0], clusters])
                    
            for cl in np.arange( clusters ):
               
                # FOR EACH CLUSTER USE THE EM ALGORITHM
                # TO CREATE THE NEW MEMBERSHIP MATRIX OF THE GAUSSIANS
                #IT IS NOT EXACTLY THE MEMBERSHIP BECAUSE IT IS
                # NORMALIZED  AFTER THIS FUNCTION ENDS
                mCl, piCl, logproba = self.calcBerPar( X, 
                                                       members[:,cl])
                   
                logprobaMatrix[:,cl] = logproba 
                
               
                means.append( mCl )
                pis.append( piCl )
                
                
           
            
            #find the maximum class log likelihood 
            #for each data point
            maxLog = np.max(logprobaMatrix, axis = 1)
            #regularization  of the log likelihood matrix
            logprobaMatrix = ( logprobaMatrix.T - maxLog).T
            probMat = np.exp( logprobaMatrix ) + regk
            sumRel = np.sum( probMat, axis = 1)
            probMat = (probMat.T / sumRel).T
            probMat = probMat*np.array(pis)
            
            params = { 'means': means, 'pis' : pis, 
                          'probMat':probMat}
            
            return params        
        
        
###calcBerPar UNDER CONSTRUCTION WORKING WITH BINARY DATA NOT SUFFICIENT TESTED
    def calcBerPar(self, X, memb):
        #CALCULATES PARAMETERS FOR EACH Bernulli product
        #FOR EACH CLUSTER
        #RETURNS:
       
        #meank : mean vector of Bernulli of class k
        #pk: mixing coefficient of Bernulli of class k
        
        #proba: the posterior probabilities, i.e probabilities of being
        #in class k given X 
        
        Nk = np.sum(memb)
        N = X.shape[0]
        #mixing coefficient
        pk = Nk/N  
        #print("Minimum of memb {} max{}".format(np.min( memb), np.max( memb)))
        meank = np.sum( ( X.T * memb ).T, axis = 0) / Nk +10**(-6)
       
        meankOne = 1-meank  + 10**(-6)
        meanklog = np.log(meank)
        meankOnelog = np.log(meankOne)
        
        #full covarinace
        logProbTerm1 = np.sum( X * meanklog, axis = 1 )
        logProbTerm2 = np.sum( (1-X) * meankOnelog, axis = 1)
            
       
        
        
        logproba = logProbTerm1 + logProbTerm2 
       #print( logproba )
      
       # print("min log prob {}, max of log prob {}".format( np.min(proba ),
       #np.max(proba )))
        #proba = model.cdf(X)*pk
        return  meank, pk, logproba            
