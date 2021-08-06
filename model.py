# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:37:22 2021

@author: J Xin

Provides implementations for ML algorithms

 
"""

import pandas as pd
import numpy as np
import os
pd.set_option('display.max_columns', 10)
import matplotlib.pyplot as plt
import random
import math
import requests
from datetime import datetime, date, timedelta
import statsmodels.api as sm
# machine learning
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, plot_roc_curve,roc_curve, auc
import xgboost as xgb
# my pypackage
from my_pypackage import config
import warnings
warnings.filterwarnings("ignore")

MODELS = {"SVM": SVC, "KNeighbors": KNeighborsClassifier, "GaussianNB": GaussianNB, 
          "RandomForest": RandomForestClassifier, 'XGBoost': xgb.XGBClassifier}
MODEL_KWARGS = {x: config.__dict__[f"{x}_PARAMS"] for x in MODELS.keys()}



class MLAgent:
    """Provides implementations for ML algorithms

    Attributes
    ----------

    Methods
    -------
        ml_prediction()
            make a prediction in a test dataset and get results
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    @staticmethod
    def ml_prediction(model, test_obs):
        """make a prediction"""
        observation = np.array(test_obs)
        try: 
            test_signal = model.predict(observation)
        except ValueError:
            test_signal = [0] * len(test_obs)   
        # return the probability of positive returns
        try:
            test_prob = model.predict_proba(observation)[:, 1]
        except ValueError or AttributeError:
            test_prob = [np.nan] * len(test_obs)        
        return test_signal, test_prob
    
    @staticmethod
    def get_model_name(**model_kwargs):
        li = []
        for key, value in model_kwargs.items():
            li.append('='.join((str(key), str(value))))
        model_name = '_'.join(tuple(li))
        
        return model_name
    
    def get_model(self, ml_method, model_kwargs=None):
        if ml_method not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[ml_method]

        print(model_kwargs)
        if ml_method in ['SVM']:
            model_kwargs['probability'] = True
        # if ml_method in ['XGBoost']:
        #     model_kwargs['use_label_encoder'] = False
        model = MODELS[ml_method](**model_kwargs).fit(self.X, self.y)
        return model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    