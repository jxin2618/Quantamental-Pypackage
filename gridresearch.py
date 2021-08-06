# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:50:13 2021

@author: J Xin

"""

import pandas as pd
import os
from my_pypackage.model import MLAgent
from my_pypackage.performance import Model_Performance
from my_pypackage.pathcreater import PathCreater
from my_pypackage import config

strategy_turnover = {'strategy1': 'daily', 'strategy2': 'weekly', 'strategy3': 'daily'} 

class GridSearch(object):
    
    def __init__(self, sec_code: str, df, df_name: str, container: dict, model_info_kw: dict):
        self.sec_code = sec_code
        self.df = df # dataset to be tested
        self.df_name = df_name
        self.container = container
        self.model_info_kw = model_info_kw 
        # self.strategy = strategy 
        # self.method = method
        # self.kernel = kernel
        
        
    def _parent_path(self):
        kwargs = self.model_info_kw.copy()
        kwargs['security'] = self.sec_code  
        
        return PathCreater.parent_dir_tuple(**kwargs)
        
    def _prediction(self, df_observations, model, model_name, signal_dir):
        """
        

        Parameters
        ----------
        df : DataFrame
            index: 'date'
            column list = [features, 'open', 'return', 'label'].
        model_name : TYPE
            DESCRIPTION.
        model : TYPE
            DESCRIPTION.
        signal_dir : TYPE
            DESCRIPTION.

        Returns
        -------
        res : DataFrame
            index: 'date'
            column list = ['signal', 'probability'].

        """
        df_copy = df_observations.copy() 
        # predictions: get signals
        X = df_copy.drop(columns=['open', 'return', 'label'], axis=1).values
        signal, prob = MLAgent.ml_prediction(model, X)
        res = pd.DataFrame(data={'signal': signal, 'probability': prob}, index=df_copy.index)
        os.chdir(signal_dir)
        res.to_excel(model_name + '.xlsx')    
        
        return res
    
    def _evaluation(self, signal, model_name, ret_dir, fig_dir): 
        ## model evaluation
        os.chdir(fig_dir)
        signal_copy = signal.copy()
        model_performance = Model_Performance(self.sec_code, signal_copy, model_name, strategy_turnover[self.model_info_kw['strategy']])
        ret, stat = model_performance.run()
        os.chdir(ret_dir)
        ret.to_excel(model_name + '_rtn.xlsx')    
        
        return stat
    
    def _get_model_performance(self, df_observations, model, model_name, signal_dir, ret_dir, fig_dir):
        signal = self._prediction(df_observations, model, model_name, signal_dir)
        stat = self._evaluation(signal, model_name, ret_dir, fig_dir)
        
        return stat
        
    def run_grid_search(self):
        parent_path = self._parent_path()
        raw_dir = PathCreater.create_raw_data_path(parent_path)
        os.chdir(raw_dir)
        self.df.to_excel('df_{}.xlsx'.format(self.df_name))
        
        signal_dir = PathCreater.create_signal_path(parent_path, self.df_name)
        ret_dir = PathCreater.create_return_path(parent_path, self.df_name)
        fig_dir = PathCreater.create_figs_path(parent_path, self.df_name)
        grid_search = pd.DataFrame()
        for key, value in self.container.items():
            stat = self._get_model_performance(self.df, value, key, signal_dir, ret_dir, fig_dir)
            grid_search = grid_search.append(stat)
            
        grid_search_dir = PathCreater.create_grid_search_path(parent_path)
        os.chdir(grid_search_dir)
        grid_search.to_excel('grid_search.xlsx', index=True)
        
        return grid_search
    
    
