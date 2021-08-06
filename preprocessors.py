# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:18:47 2021

@author: J Xin
"""
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import sys
sys.path.append('D:/Xin/Program/AI_Quantamental/')
from my_pypackage import config


class FeatureEngineer:
    """ 
    """
    
    
    def __init__(self, df):

        self.df = df
   
    def normalize_data(self):
        """main method to do the feature engineering
        """
        df = self.df.copy()
        quantile_99 = df.expanding(min_periods=180).quantile(0.99)
        quantile_01 = df.expanding(min_periods=180).quantile(0.01)
        df[df > quantile_99] = quantile_99
        df[df < quantile_01] = quantile_01
        data_mean = df.expanding(min_periods=180).mean()
        data_std = df.expanding(min_periods=180).std()
        data_normal = (df - data_mean) / data_std
        res = data_normal.iloc[180:, :]
        
        return res
        
        
    @staticmethod
    def data_split(df, start: str, end: str):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df.index.astype(str) >= start) & (df.index.astype(str) <= end)]
        data = data.sort_index(ascending=True)
        
        return data
    
    @staticmethod
    def add_user_defined_feature(data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """          
        df = data.copy()
        df['return'] = df.open.pct_change(1)
        df['return_lag_1'] = df.open.pct_change(2)
        df['return_lag_2']=df.open.pct_change(3)
        #df['return_lag_3']=df.close.pct_change(4)
        #df['return_lag_4']=df.close.pct_change(5)
        return df
    
    @staticmethod
    def resample_to_weekly_price(df, weekday: str) -> pd.DataFrame:
        data = df.copy()
        rule = '-'.join(('W', weekday))
        data = data.resample(rule, label='left', closed='left').first().fillna(method='ffill')
        return data
    
    @staticmethod
    def resample_to_weekly_factors(df, weekday: str):
        data = df.copy()
        rule = '-'.join(('W', weekday))
        data = data.resample(rule, label='right', closed='right').last().fillna(method='ffill')  
        return data
    
    @staticmethod
    def resample_to_business_day(df):
        data = df.copy()
        data = data.resample('B').last()
        
        return data

    
    
    
    
    
    
    
    
    
    
    