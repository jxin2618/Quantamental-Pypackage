# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:36:45 2021

@author: J Xin

Store the parameters of ML algorithms including SVM, KNN, Naive Bayes, Random Forest, XGBoost
Store the root path to save the results

"""


# path

parent_dir = 'D:/Xin/Program/AI_Quantamental/results/'


## Model Parameters
SVM_PARAMS = {'kernel': 'linear',
			  'C': 0.1,
			  'gamma': 0.1,
              'degree': 2,
              }

KNeighbors_PARAMS = {'n_neighbors': 10,
			  'weights': 'distance',
              'p': 1
              }

GaussianNB_PARAMS = {'var_smoothing' : 0.01
              }

RandomForest_PARAMS = {'n_estimators' : 50,
                       'max_features': 50,
                       'criterion': 'gini',
                       'random_state': 42
              }

XGBoost_PARAMS = {'n_estimator': 50,
                  'max_depth' : 50,
                  'learning_rate': 0.1,
                  'random_state': 42
            }




