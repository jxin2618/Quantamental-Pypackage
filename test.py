# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:01:11 2021

@author: J Xin
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('D:/Xin/Program/AI_Quantamental/')
import my_pypackage
from my_pypackage.datadownloader import DataDownloader
from my_pypackage.preprocessors import FeatureEngineer
from my_pypackage.strategy import Strategy
from my_pypackage.model import MLAgent
from my_pypackage.gridresearch import GridSearch

data_downloader_I = DataDownloader('2015-01-01', '2021-04-30', 'I.CCRI')
data_downloader_RB = DataDownloader('2015-01-01', '2021-04-30', 'RB.CCRI')
raw_I = data_downloader_I.fetch_data()
raw_RB = data_downloader_RB.fetch_data()
price_I = data_downloader_I.fetch_open_price()
price_RB = data_downloader_RB.fetch_open_price()

normal_I = FeatureEngineer(raw_I).normalize_data()
normal_RB = FeatureEngineer(raw_RB).normalize_data()

#%%
Strategy_I = Strategy(normal_I, price_I)
I_df1 = Strategy_I.strategy1()
I_df2 = Strategy_I.strategy2()
I_df3 = Strategy_I.strategy3()

Strategy_RB = Strategy(normal_RB, price_RB)
RB_df1 = Strategy_RB.strategy1()
RB_df2 = Strategy_RB.strategy2()
RB_df3 = Strategy_RB.strategy3()


#%%
train_set_1 = FeatureEngineer.data_split(I_df1, '2015-07-01', '2018-12-31')
cross_validation_set_1 = FeatureEngineer.data_split(I_df1, '2019-01-01', '2019-12-31')
test_set_1 = FeatureEngineer.data_split(I_df1, '2020-01-01', '2021-12-31')
train_set_1  = train_set_1.loc[train_set_1.loc[:, 'label'] != 0, :]


X_train = train_set_1.drop(columns=['open', 'return', 'label'], axis=1).fillna(0).values
y_train = train_set_1['label'].fillna(0).values
cross_validation_set_1 = cross_validation_set_1.fillna(0)
test_set_1 = test_set_1.fillna(0)

#%% SVM

C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
gamma_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
degree_list = [2,3]

gaussian_model_container = {}
poly_model_container = {}
linear_model_container = {}



# linear kernel
print('linear kernel')
for c in C_list:
    model_kwargs = {'kernel': 'linear', 'C': c}
    model_name = MLAgent.get_model_name(**model_kwargs)
    clf = MLAgent(X_train, y_train).get_model(ml_method='SVM', model_kwargs=model_kwargs)
    linear_model_container[model_name] = clf
        
# gaussian kernel
print('gaussian kernel')
for c in C_list:
    for g in gamma_list:
        model_kwargs = {'kernel': 'rbf', 'C': c, 'gamma': g}
        model_name = MLAgent.get_model_name(**model_kwargs)
        clf = MLAgent(X_train, y_train).get_model(ml_method='SVM', model_kwargs=model_kwargs)
        gaussian_model_container[model_name] = clf
    
# polynomial kernel
print('polynomial kernel')
for c in C_list:
    for g in gamma_list:
        for d in degree_list:
            model_kwargs = {'kernel': 'poly', 'C': c, 'gamma': g, 'degree': d}
            model_name = MLAgent.get_model_name(**model_kwargs)
            clf = MLAgent(X_train, y_train).get_model(ml_method='SVM', model_kwargs=model_kwargs)
            poly_model_container[model_name] = clf


def main(container, kernel):
    grid_search = GridSearch(sec_code='I.CCRI', cv_df=cross_validation_set_1, test_df=test_set_1, container=linear_model_container, method='SVM', kernel='linear')
    cv_grid_search, test_grid_search = grid_search.save_grid_search_data()
    
    return cv_grid_search, test_grid_search

linear_cv, linear_test = main(linear_model_container, 'Linear')

















