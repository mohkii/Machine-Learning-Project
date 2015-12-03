# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:19:21 2015

@author: hannahjin
"""
import pandas as pd
from sklearn import svm
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

# os.path.dirname(os.path.realpath('__file__')) current path
trainFile = '../train.csv'
testFile = '../test.csv'
train_data = pd.read_csv(trainFile, sep=',', header=0)
# test_data = pd.read_csv(trainFile, sep=',', header=0)
train_data.fillna(0, inplace=True)

sample_train = train_data.ix[:1000,:]
sample_test = train_data.ix[5000:6000]
train_data = sample_train
test_data = sample_test
# colNames = train_data.columns.values.tolist()
time_series_train = train_data.filter(regex='Ret_\d')
return_features = pd.concat([time_series_train.ix[:, :119].mean(axis=1), 
                             time_series_train.ix[:, :119].var(axis=1)], axis=1)
time_series_test = test_data.filter(regex='Ret_\d')*10000
return_test = pd.concat([time_series_test.ix[:, :119].mean(axis=1), 
                         time_series_test.ix[:, :119].var(axis=1)], axis=1)
# 'labels'
targets = pd.concat([time_series_train.ix[:, 119:].mean(axis=1), 
                     time_series_train.ix[:, 119:].var(axis=1)], axis=1)
return_features.columns = ['mean', 'variance']
return_test.columns = ['mean', 'variance']
targets.columns = ['mean', 'variance']

train_data.ix[:, :27]

svmclf = svm.SVR()
svmclf.fit(pd.concat([train_data.ix[:, :27], return_features], axis=1), targets['mean']) 
pred1 = svmclf.predict(pd.concat([test_data.ix[:, :27], return_test], axis=1)) 
mean_squared_error1 = mse(targets['mean'], pred1)

bayesRidge = linear_model.BayesianRidge()
bayesRidge.fit(pd.concat([train_data.ix[:, :27], return_features], axis=1), targets['mean'])
pred2 = bayesRidge.predict(pd.concat([test_data.ix[:, :27], return_test], axis=1)) 
mean_squared_error2 = mse(targets['mean'], pred2) # 2.0655e-08

lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(pd.concat([train_data.ix[:, :27], return_features], axis=1), targets['mean'])
pred3 = bayesRidge.predict(pd.concat([test_data.ix[:, :27], return_test], axis=1)) 
mean_squared_error3 = mse(targets['mean'], pred3) # 2.0655e-08


