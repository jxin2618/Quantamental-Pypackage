# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:55:55 2021

@author: J Xin
"""
import numpy as np
import pandas as pd
import empyrical as ep
import matplotlib.pyplot as plt
import requests
from datetime import timedelta
from sklearn.metrics import roc_auc_score, plot_roc_curve,roc_curve, auc, accuracy_score
from my_pypackage.datadownloader import DataDownloader
from my_pypackage.preprocessors import FeatureEngineer

class Model_Performance(object):
        
    def __init__(self, sec_code, signal_df, model_name, period):
        """
        

        Parameters
        ----------
        sec_code : TYPE
            DESCRIPTION.
        signal_df : DataFrame
            columns list = ['date', 'signal', 'probability']
        model_name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sec_code = sec_code
        self.signal_df = signal_df
        self.model_name = model_name
        self.period = period
    
    def _get_open_price(self):
        signal_df = self.signal_df.reset_index()
        signal_df = signal_df.sort_values(by='date',ascending=True)
        start_date = pd.Timestamp(signal_df['date'].values[0])
        end_date = pd.Timestamp(signal_df['date'].values[-1]) + timedelta(days=7)
        downloader = DataDownloader(str(start_date), str(end_date), self.sec_code)
        res = downloader.fetch_open_price()
        
        return res
    
    def _get_signal_merged_with_price(self):
        # signal = self.signal_df[['signal']]
        signal = self.signal_df.copy()
        # signal = signal.set_index('date')
        signal = FeatureEngineer.resample_to_business_day(signal)
        price = self._get_open_price()
        merged = pd.merge(signal, price, on='date', how='outer').fillna(method='ffill')
        merged = merged.sort_values(by='date', ascending=True) 
        
        return merged
    
    @classmethod
    def get_position(self, data):
        """
        

        Parameters
        ----------
        data : DataFrame
            index = 'date'
            column list = ['signal', 'probability', 'open'].

        Returns
        -------
        None.

        """
        
        df = data.copy()
        df['position'] = df['signal'].shift(1)
        
        return df
    
    @classmethod
    def get_daily_return(self, data):
        """
        

        Parameters
        ----------
        data : DataFrame
            index = 'date'
            colum list = ['siganl', 'probability', 'open', 'position'].

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        
        df = data.copy()
        df.loc[:, 'price_change_pct'] = df.loc[:, 'open'].pct_change()
        df.loc[:, 'turnover'] = df['position'].diff().fillna(0)
        df.loc[:, 'daily_return_before_cost'] = df.price_change_pct * df['position'].shift(1).fillna(0)
        df.loc[:, 'daily_return'] = (df.loc[:, 'daily_return_before_cost'] - df.loc[:, 'turnover'].abs() * 0.0003).fillna(0)
        # df.pop('turnover')
        return df
    
    def BackTestStats(self, data):
        df = data.copy()
        df_copy = df.copy()
        df_copy = df_copy.loc[df_copy.loc[:, 'daily_return'] != 0, :]
  
        sharpe = ep.sharpe_ratio(df['daily_return'].values, period=self.period)
        annual_return = ep.annual_return(df['daily_return'].values, period=self.period)
        annual_volatility = ep.annual_volatility(df['daily_return'].values, period=self.period)
        win_ratio = (np.sign(df['daily_return']) == 1).sum() / len(df['daily_return'].values) 
        
        auc_score = roc_auc_score(np.sign(df_copy['daily_return'].values)[1:], df_copy['probability'].fillna(0).shift(1).values[1:], average=None)
            
        max_drawdown = ep.max_drawdown(df['daily_return'].values)
        stats = {'Annual return': [annual_return], 'Annual volatility': [annual_volatility],
                 'Sharpe ratio': [sharpe], 'Max drawdown': [max_drawdown], 
                 'Win Rate': [win_ratio], 'AUC': [auc_score]}
        res = pd.DataFrame(stats)
        res.index = [self.model_name]
    
        return res
    
    def BackTestPlot(self, df):
        df.loc[:, 'net_value'] = (1 + df.loc[:, 'daily_return'].values).cumprod()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = df.index
        y = df.loc[:, 'net_value']
        w = df.loc[:, 'signal']
        ax1.plot(x, y, color='r', label='net_value')
        ax2 = ax1.twinx()
        ax2.bar(x, w, color='orange', label='position')
        plt.xlabel('date')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title('_'.join((self.model_name, 'positions')))
        plt.grid()
        plt.savefig('.'.join((self.model_name + '_positions', 'png')))
        plt.close()
        # plt.show()
        
        return
    
    def run(self):
        df = self._get_signal_merged_with_price()
        position = self.get_position(df)
        ret = self.get_daily_return(position)
        stat = self.BackTestStats(ret)
        self.BackTestPlot(ret)
        
        return ret, stat
    