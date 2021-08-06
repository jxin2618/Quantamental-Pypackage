# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:52:09 2021

@author: J Xin
"""

import os
from my_pypackage import config

class PathCreater(object):

    @classmethod
    def _tuple_to_dirs(cls, path_comp: tuple)->str:
        length = len(path_comp)
        for i in range(1, length):
            path =  "/".join((path_comp[0: i + 1]))       
            if not os.path.exists(path):
                os.makedirs(path)
        
        return path
    
    @classmethod
    def parent_dir_tuple(cls, **kwargs):
        res = ()
        for i in ['security', 'strategy', 'method', 'kernel', 'weight']:
            iv = kwargs.get(i)
            if iv:
                res = res + (iv, )
        # if self.kernel is not None:
        #     res = (config.parent_dir, self.security, self.strategy, self.method, self.kernel)
        # else:
        #     res = (config.parent_dir, self.security, self.strategy, self.method)
            
        return  (config.parent_dir, ) + res
    
    @classmethod
    def create_raw_data_path(cls, parent_dir):
        path_tuple = parent_dir + ('data', 'raw_data', )
        path = cls._tuple_to_dirs(path_tuple) 
        
        return path
    
    @classmethod
    def create_signal_path(cls, parent_dir, data_set_name):
        path_tuple = parent_dir + ('data', 'intermediate_data', 'signal', data_set_name) 
        path = cls._tuple_to_dirs(path_tuple) 

        return path
    
    @classmethod
    def create_return_path(cls, parent_dir, data_set_name):
        path_tuple = parent_dir + ('data', 'intermediate_data', 'return', data_set_name)
        path = cls._tuple_to_dirs(path_tuple) 

        return path
    
    @classmethod
    def create_figs_path(cls, parent_dir, data_set_name):
        path_tuple = parent_dir + ('figure', 'signal', data_set_name) 
        path = cls._tuple_to_dirs(path_tuple) 
        
        return path
    
    @classmethod
    def create_grid_search_path(cls, parent_dir):
        path_grid_search_tuple = parent_dir + ('data', 'output_data', 'grid_research')
        grid_search_dir = cls._tuple_to_dirs(path_grid_search_tuple) 

        return grid_search_dir
    
    
    