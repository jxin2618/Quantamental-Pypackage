# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:12:00 2021

@author: J Xin
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('D:/Xin/Program/AI_Quantamental/')
from my_pypackage.datadownloader import DataDownloader
from my_pypackage.preprocessors import FeatureEngineer
from my_pypackage.strategy import Strategy
from my_pypackage.model import MLAgent
from my_pypackage.gridresearch import GridSearch
from my_pypackage.pathcreater import PathCreater 
from operator import methodcaller


#%% Step0 : Initialize Parameters

################### Date ####################
start_date = '2015-01-01'
end_date = '2021-04-30'
start_train_date = '2015-07-01'
end_train_date = '2018-12-31'
start_cross_validation_date = '2019-01-01'
end_cross_validation_date = '2019-12-31'
start_test_date = '2020-01-01'
end_test_date = '2021-12-31'

############## Model Parameters ############
C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10] # SVM---Regularization parameter.
gamma_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]#SVM---Kernel coefficient 
degree_list = [2,3] # SVM---Degree of the polynomial kernel function 
var_list = [1e-09, 5e-09, 1e-08, 5e-08, 1e-07, 5e-07,
            1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 
            1e-03, 5e-03, 1e-02, 5e-02, 1e-01, 5e-01] # Naive_Bayes---Portion of the largest variance
n_neighbors_list = np.arange(5, 105, 5) # K Neighbors---Number of neighbors 
n_tree_list = [50] + list(np.arange(100, 600, 100)) # Random Forest
max_feature_list = [50] + list(np.arange(100, 600, 100)) # Random Forest
max_depth_list =  list(np.arange(3, 10, 1)) # xgboost
################ Subject ######################

#%%
class DataPrepare(object):
    def __init__(self, sec_code, strategy):
        self.sec_code = sec_code
        self.strategy = strategy
            
    def _download_data(self):
        data_downloader = DataDownloader(start_date, end_date, self.sec_code)
        raw_data = data_downloader.fetch_data()
        price = data_downloader.fetch_open_price()
        return raw_data, price
    
    def _feature_engineering(self):
        raw, price = self._download_data()
        normal = FeatureEngineer(raw).normalize_data()
        return normal, price
    
    def _get_strategy(self):
        normal, price = self._feature_engineering()
        strategy_class = Strategy(normal, price)
        df = methodcaller(self.strategy)(strategy_class)
        
        return df
    
    def get_data(self):
        df = self._get_strategy()
        train_set = FeatureEngineer.data_split(df, start_train_date, end_train_date)
        cross_validation_set = FeatureEngineer.data_split(df, start_cross_validation_date, end_cross_validation_date)
        cross_validation_set = cross_validation_set.fillna(0)
        test_set = FeatureEngineer.data_split(df, start_test_date, end_test_date)
        test_set = test_set.fillna(0)
        train_set  = train_set.loc[train_set.loc[:, 'label'] != 0, :]
        
        return train_set, cross_validation_set, test_set
    
    
#%%  
class Train(object):
    
    def __init__(self, sec_code, strategy, train_set, cross_validation_set, test_set):
        self.sec_code = sec_code
        self.strategy = strategy
        self.train_set = train_set
        self.cross_validation_set = cross_validation_set
        self.test_set = test_set
        
    def _parent_path(self, model_info_kw):
        kwargs = model_info_kw.copy()
        kwargs['security'] = self.sec_code  
        
        return PathCreater.parent_dir_tuple(**kwargs)
    
    def _save_train_data(self, model_info_kw):
        parent_path = self._parent_path(model_info_kw)
        raw_dir = PathCreater.create_raw_data_path(parent_path)
        os.chdir(raw_dir)
        self.train_set.to_excel('df_{}.xlsx'.format('train'))
       
        return
        
    def _get_obs_n_target(self):
        X_train = self.train_set.drop(columns=['open', 'return', 'label'], axis=1).fillna(0).values
        y_train = self.train_set['label'].values
        
        return X_train, y_train
    
    def grid_search(self, model_container, model_info_kw):
        self._save_train_data(model_info_kw)
        cross_validation_grid_search = GridSearch(sec_code=self.sec_code, df=self.cross_validation_set, df_name='cross_validation', container=model_container, model_info_kw=model_info_kw).run_grid_search()
        test_grid_search = GridSearch(sec_code=self.sec_code, df=self.test_set, df_name='test', container=model_container, model_info_kw=model_info_kw).run_grid_search()
        cv_tmp = cross_validation_grid_search.loc[:, ['Annual return', 'Sharpe ratio', 'Win Rate', 'AUC']]
        test_tmp = test_grid_search.loc[:, ['Annual return', 'Sharpe ratio', 'Win Rate', 'AUC']] 
        cv_tmp = cv_tmp.add_prefix('CV_')
        test_tmp = test_tmp.add_prefix('Test_')
        order = ['CV_Annual return', 'Test_Annual return', 'CV_Sharpe ratio', 'Test_Sharpe ratio',
                 'CV_Win Rate', 'Test_Win Rate', 'CV_AUC', 'Test_AUC']  
        cv_and_test = pd.concat([cv_tmp, test_tmp], axis=1)
        cv_and_test = cv_and_test[order]
        
        return cv_and_test 
        

    def train_svm(self):
        X_train, y_train = self._get_obs_n_target()
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
        
        linear_cv_and_test  = self.grid_search(model_container=linear_model_container, model_info_kw = {'strategy' : self.strategy, 'method': 'SVM', 'kernel': 'linear'})
        gauss_cv_and_test  = self.grid_search(model_container=gaussian_model_container, model_info_kw = {'strategy' : self.strategy, 'method': 'SVM', 'kernel': 'gauss'})
        poly_cv_and_test  = self.grid_search(model_container=poly_model_container, model_info_kw = {'strategy' : self.strategy, 'method': 'SVM', 'kernel': 'poly'})
        
        return linear_cv_and_test, gauss_cv_and_test, poly_cv_and_test
    
    def train_naivebayes(self):
        X_train, y_train = self._get_obs_n_target()
        nb_model_container = {}
        for var in var_list:
            model_kwargs = {'var_smoothing': var}
            model_name = MLAgent.get_model_name(**model_kwargs)
            clf = MLAgent(X_train, y_train).get_model(ml_method='GaussianNB', model_kwargs=model_kwargs)
            nb_model_container[model_name] = clf
        nb_cv_and_test  = self.grid_search(model_container=nb_model_container, model_info_kw = {'strategy' : self.strategy, 'method': 'GaussianNB'})
        return nb_cv_and_test
        
    
    def train_kneighbors(self):
        X_train, y_train = self._get_obs_n_target()
        uniform_model_container = {}
        distance_model_container = {}
        if self.strategy == 'strategy2':
            n_neighbors_list = np.arange(5, 55, 5) 
        else:
            n_neighbors_list = np.arange(5, 105, 5) 
        print(n_neighbors_list)
        for n in n_neighbors_list:
            for p in [1, 2]:
                model_kwargs = {'weights': 'uniform', 'n_neighbors': n, 'p': p}
                model_name = MLAgent.get_model_name(**model_kwargs)
                clf = MLAgent(X_train, y_train).get_model(ml_method='KNeighbors', model_kwargs=model_kwargs)
                uniform_model_container[model_name] = clf
            uniform_cv_and_test  = self.grid_search(model_container=uniform_model_container, model_info_kw = {'strategy' : self.strategy, 'method': 'KNeighbors', 'weight': 'uniform'})
        for n in n_neighbors_list:
            for p in [1, 2]:
                model_kwargs = {'weights': 'distance', 'n_neighbors': n, 'p': p}
                model_name = MLAgent.get_model_name(**model_kwargs)
                clf = MLAgent(X_train, y_train).get_model(ml_method='KNeighbors', model_kwargs=model_kwargs)
                distance_model_container[model_name] = clf
            distance_cv_and_test  = self.grid_search(model_container=uniform_model_container, model_info_kw = {'strategy' : self.strategy, 'method': 'KNeighbors', 'weight': 'distance'})

        return uniform_cv_and_test, distance_cv_and_test    
       
    def train_randomforest(self):
        X_train, y_train = self._get_obs_n_target()
        randomforest_model_container = {}
        for n in n_tree_list:
            for features in max_feature_list:
                model_kwargs = {'n_estimators': n, 'max_features': features}
                model_name = MLAgent.get_model_name(**model_kwargs)
                clf = MLAgent(X_train, y_train).get_model(ml_method='RandomForest', model_kwargs=model_kwargs)
                randomforest_model_container[model_name] = clf
                randomforest_cv_and_test  = self.grid_search(model_container=randomforest_model_container, model_info_kw = {'strategy' : self.strategy, 'method': 'RandomForest'})

        return randomforest_cv_and_test
    
    def train_xgboost(self):
        X_train, y_train = self._get_obs_n_target()
        xgboost_model_container = {}
        for n in n_tree_list:
            for d in max_depth_list:
                model_kwargs = {'n_estimators': n, 'max_depth': d}
                model_name = MLAgent.get_model_name(**model_kwargs)
                clf = MLAgent(X_train, y_train).get_model(ml_method='XGBoost', model_kwargs=model_kwargs)
                xgboost_model_container[model_name] = clf
                xgboost_cv_and_test  = self.grid_search(model_container=xgboost_model_container, model_info_kw = {'strategy' : self.strategy, 'method': 'XGBoost'})

        return xgboost_cv_and_test
    



