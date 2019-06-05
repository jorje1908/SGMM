




import sys

sys.path.append('../SGMM')
sys.path.append('../metrics')
sys.path.append('../loaders')
sys.path.append('../oldCode')
sys.path.append('../visual')
sys.path.append('../testingCodes')
sys.path.append('../otherModels')
sys.path.append('../experiments')
#sys.path.append('../oldCode')

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import multivariate_normal
from supervisedGmm import SupervisedGMM
from metricsFunctions import optimalTau, calc_metrics, metrics_cluster, sgmmResults
import time
import pandas as pd
from dataGen import genData1D, bayeOpt

np.random.seed( seed = 0 )





#INITIALIZING THE GAUSSIAN PARAMETERS
covG = np.array( 3)
covG2 = np.array( 3 )
enhance = 100

mix = [0.5, 0.5]
#GAUSS 1
m1 = [ 0 ]
cov1 = covG
g1 = multivariate_normal(mean = m1, cov = cov1)

#GAUSS 2
m2 = [ 3 ]
cov2 = covG
g2 = multivariate_normal(mean = m2, cov = cov2)

#GAUSS 3
m3 = [ 0 ]
cov3 = covG2
g3 = multivariate_normal(mean = m3, cov = cov3)

#creating the separting hyperplane based  on points p1 p2 for third gaussian
b3 = 0
w3 = np.array([-b3, 1])*enhance

#GAUSS 4
m4 = [ 3 ]
cov4 = covG2
g4 = multivariate_normal(mean = m4, cov = cov4)

#creating the separting hyperplane based  on points p1 p2 for third gaussian

b4 =  3
w4 = np.array([-b4, -1])*enhance


#opttimal bayesian Error

err, per, deb = bayeOpt( mix[0], mix[1], m1, m2, m3, m4, cov1, cov2,
                                                         cov3, cov4, w3, w4, 40000 )

#DATA VISUALIZATION
gen =  genData1D( mix[0], mix[1], m1, m2, m3, m4, cov1, cov2,
                                                         cov3, cov4, w3, w4, 12000 )
scale = 0.03
genX = gen[ :, 0:2 ]
genY = gen[ :, 2 ]
indPos = np.where( genY == 1 )[0]
indNeg = np.where( genY == 0 )[0]

fig, ax = plt.subplots(1,1, figsize = [8, 8])
ax.scatter( genX[indPos, 1], np.zeros_like(genX[indPos, 1]), s = scale)
ax.scatter( genX[indNeg, 1],  np.zeros_like(genX[indNeg, 1]), s = scale)
ax.plot(np.ones(2)*b3, [-1, 1])
ax.plot(np.ones(2)*b4, [-1, 1])
ax.legend(['Pos', 'Neg','Dec1', 'Dec4'])
#fig.savefig('Results/note1/genD.png', bbox_inches = 'tight')





#MODEL INITIALIZATION
adaR = 1  #adaptive regularization
alpha = [0.01] #regularization parameter
n_clusters = 2 #number of clusters
vrb = 0  #verbose output
cv = 10 #cross validation splits
scoring = 'neg_log_loss' #scoring negative log loss
mcov = 'diag' #diagonal covariance
mx_it = 1000  #maximum training epochs
mx_it2 = 10 #maximum EM iterations
warm = 0    #warm iteration
km = 1      #kmeans initialization
mod = 1     #mod zero or one 
model = SupervisedGMM(  n_clusters = n_clusters, max_iter2 = mx_it2, tol = 10**(-10),
                         max_iter = mx_it, alpha = alpha, mcov = mcov, adaR = adaR,
                         transduction = 1, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol2 = 10**(-2) )





#SETTING  REPETITIVE PARAMETERS
points = 100 #points for training 
averaging = 1 #HOW MANY TIMES TO RUN AVERAGING
start = 2       #TEST POINTS START
end = 4     #TEST POINT END
step = 2        #step of test points
test1 = []      #lists to keep accuracy of test used in training
test2 = []      #lists to keep accuracy of fresh batch of tests
meansOut = []   #SAVING ALL MEANS
weightsOut = [] #SAVING ALL WEIGHTS

for n in np.arange( start, end, step ): #
    
    N = points + n         #POINTS FOR TRAINING
    N1 = n              #POINTS FOR TESTING
    split = 1 - points/N   #TRAIN - TEST SPLIT
    testMets1 = 0
    testMets2 = 0
    meansIn = [] #SAVING THE MEANS
    weightsIn = [] #SAVING WEIGHTS
    for i in np.arange( averaging ):
        
        print("ITERATION OF AVERAGING :{}, batch: {} end batch:{}".format( i, n, end ))
        
        #DATA GENERATION
        data = genData1D( mix[0], mix[1], m1, m2, m3, m4, cov1, cov2,
                                                         cov3, cov4, w3, w4, N )
        #FRESH DATA GENERATION
        indTest = genData1D( mix[0], mix[1], m1, m2, m3, m4, cov1, cov2,
                                                        cov3, cov4, w3, w4, N1 )
        #TRAIN DATA
        X = data[:, 0:2]
        Y = data[:, 2] 
        
        #FRESH DATA FOR EVALUATION
        Xind = indTest[:,0:2]
        Yind = indTest[:, 2 ]
        
        #SPLIT IN TRAIN TEST DATA
        Xtrain, Xtest, ytrain, ytest = model.split( X = X, y = Y, split = split )
        
        #FIT THE MODEL
        model = model.fit( Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain, kmeans = km,
                          ind2 = [1], mod = mod )

        #PREDICT THE INTERNAL PROBABILITIES 
        probTest, probTrain = model.predict_prob_int( Xtest = Xtest, Xtrain = Xtrain )
        #PREDICT  PROBABILITIES FOR FRESH TEST
        probTest2 = model.predict_proba( Xind )

        res = sgmmResults( model , probTest.copy(), probTrain.copy(), ytest, ytrain)
        res2 = sgmmResults( model , probTest2, probTrain.copy(), Yind, ytrain)
        
        testMets1 += res['testMet'].values
        testMets2 += res2['testMet'].values
        weightsIn.append(res['weights'])
        meansIn.append( res['means'])
     #END OF INSIDE LOOP
    meansOut.append( meansIn)
    weightsOut.append( weightsOut )
    test1.append( testMets1/averaging )
    test2.append( testMets2/averaging )





#Results Visualization
testMets1 = np.squeeze( np.array( test1 ), axis = 1 )
testMets2 = np.squeeze( np.array( test2 ), axis = 1 )

columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']


index = np.arange(  testMets1.shape[0] )
testMets1Pd = pd.DataFrame( testMets1, columns = columns)
testMets2Pd = pd.DataFrame( testMets2, columns = columns)
print(testMets1Pd)
print('#######################')
print(testMets2Pd)
    
fig, ax = plt.subplots( 1, 1)

ax.plot( index, testMets1Pd['accuracy'])
ax.plot( index, testMets2Pd['accuracy'])
ax.plot( index, testMets1Pd['auc'])
ax.plot( index, testMets2Pd['auc'])

ax.set_xlabel(' folds ')
ax.set_ylabel(' Performance_Metric ')
ax.legend( [ 'acc1', 'acc2', 'auc1', 'auc2'] )
#fig.savefig('Results/note1/resu.png', bbox_inches = 'tight')

#In[]:



