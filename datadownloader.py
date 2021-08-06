# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:01:06 2021

@author: J Xin
"""

import pandas as pd
import requests
import os

class DataDownloader:
    
    def __init__(self, start_date:str, end_date:str, tic:str):
        self.start_date = start_date
        self.end_date = end_date
        self.tic= tic

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from sql
        Parameters
        ----------
        
        Returns
        -------
        `pd.DataFrame` 
        """
        os.chdir(r'\\192.168.8.90\投研部\Jessica\test_data')
        if self.tic in ['RB.CCRI', 'HC.CCRI', 'I.CCRI', 'J.CCRI', 'JM.CCRI', 'ZC.CCRI']:
            f = pd.read_hdf('data.h5', 'snc')
        if self.tic in ['CU.CCRI', 'ZN.CCRI', 'AL.CCRI', 'NI.CCRI']:
            f = pd.read_hdf('data.h5', 'met')
        data = f.loc[f.loc[:, 'sec_code'] == self.tic, :]
        # extract I.CCRI data
        table = pd.pivot_table(data, index=['date'], columns=['factor_code'], values='factor_value')
        table = table.sort_values(by='date')
        
        return table
        
    def fetch_open_price(self):
        """
        

        Returns
        -------
        res : DataFrame
            index= 'date'
            columns = ['date', 'open']

        """
        url = 'http://192.168.7.209:6868/common'
        res = requests.post(url, json=dict(start=self.start_date, end=self.end_date, columns=['open'], sec_code=self.tic))
        res = pd.DataFrame(res.json())
        res = res.loc[:, ['date', 'open']]
        res.loc[:, 'date'] = res.loc[:, 'date'].apply(lambda x: pd.Timestamp(str(x)[0:10]))
        res= res.set_index('date')
        
        return res
    
   
    
    
    
    
    
    
    
    
    
    
