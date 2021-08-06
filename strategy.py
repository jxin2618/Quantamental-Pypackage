# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:40:02 2021

@author: J Xin

# 对齐因子序列与价格序列
#        Returns
        -------
        df : DataFrame
            columns: factor_list + ['open', 'return', 'label'].

"""

import numpy as np
import pandas as pd
from datetime import timedelta
from my_pypackage.preprocessors import FeatureEngineer


class Strategy(object):

    def __init__(self, raw_data, price):
        self.raw_data = raw_data
        self.price = price

    def strategy1(self):
        """
        

        Returns
        -------
        df : DataFrame
            columns: factor_list + ['open', 'return', 'label'].
            index: data's update_date
            'label': 当天因子对应的方向标签

        """
        # 日度换手，每个交易日收盘后获得因子，获得信号方向，下一个交易日调仓
     
        df_data = self.raw_data.copy()
        df_price = self.price.copy()

        df_price['return'] = df_price.open.pct_change(1)
        df = pd.merge(df_data, df_price, on='date', how='inner')
        df['label'] = np.sign(df['return'].shift(-2).fillna(0).values)
        
        return df
    
    def strategy2(self):
        """
        

        Returns
        -------
        df : DataFrame
            columns: factor_list + ['open', 'return', 'label'].
            index: data's update_date
            'label': 当天因子对应的方向标签

        """
        # 周度换手，对日频因子降采样，取每周五切片的因子值
        # 每周最后一个交易日收盘后获得因子，获得信号方向，下一周第一个交易日调仓
        df_data = self.raw_data.copy()
        df_price = self.price.copy()
        
        df_data = FeatureEngineer.resample_to_weekly_factors(df_data, 'FRI')
        df_price = FeatureEngineer.resample_to_weekly_price(df_price, 'MON')
        factors_date = df_data.index
        
        df = pd.merge(df_data.reset_index(), df_price.reset_index(), 
                      on='date', how='outer')
        df = df.sort_values('date', ascending=True)
        df['open'] = df['open'].fillna(method='bfill')
        df = df.loc[df['date'].isin(factors_date)]
        df['return'] = df.open.pct_change(1)
        df['label'] = np.sign(df['return'].shift(-1).fillna(0).values)
        df = df.set_index('date') 
        
        return df
    
    
    def strategy3(self):
        """
        

        Returns
        -------
        df : DataFrame
            columns: factor_list + ['open', 'return', 'label'].
            index: data's update_date
            'label': 当天因子对应的方向标签

        """
        # 日度换手
        # 用周度收益率替换训练集的日度收益率
        # 每个交易日收盘后获得因子，获得信号方向，下一个交易日调仓
        df_data = self.raw_data.copy()
        df_price = self.price.copy()
        
        df_price_resample = FeatureEngineer.resample_to_weekly_price(df_price, 'MON')
        df_price_resample['return'] = df_price_resample.open.pct_change(1)
        df_price_resample['label'] = np.sign(df_price_resample['return'].shift(-1).fillna(0).values)
        df_price_resample = df_price_resample.reset_index()
        df_price_resample['date'] = df_price_resample['date'].apply(lambda x: x - timedelta(days=1))
        
        df_price_with_label = pd.merge(df_price.reset_index(),
                              df_price_resample[['date', 'return', 'label']], how='outer').sort_values(by='date')
        df_price_with_label[['return', 'label']] = df_price_with_label[['return', 'label']].fillna(method='bfill')
        df_price_with_label = df_price_with_label.loc[df_price_with_label['date'].isin(df_price.index)]
        
        df = pd.merge(df_data.reset_index(), df_price_with_label, 
              on='date', how='inner')
        df = df.set_index('date')
        
        return df

