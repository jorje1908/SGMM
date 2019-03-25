#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:55:20 2019

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
#from sklearn.neural_network import MLPClas
from  scipy.stats import multivariate_normal

from supervisedGmm import SupervisedGMM
from metricsFunctions import calc_metrics, CalculateSoftLogReg, optimalTau
from superGmmMother import superGmmMother
from loaders2 import loader


np.random.seed( seed = 0)
data, dataS, idx = loader(4000, 300)


mother = superGmmMother( data , n_clusters = 3)

results = mother.fit_results(fitted = 1)
#model = mother.model
#params = model.fit(Xtrain =mother.Xtrain,Xtest =  mother.Xtest,
#                   ytrain = mother.ytrain)


#gmm1 = params['Gmms'][0]
#xs = gmm1.cdf(mother.Xtrain[0])