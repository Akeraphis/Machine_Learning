# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:25:50 2017

@author: JCD1
"""
#Load libraries
import os
import pandas as pd

#Load data files
path = "./Users/jcd1/Documents/GitHub/Machine_Learning/Stock_Pred/data"
directory = os.path.join("c://",path)
csv_list = []
stocks = []
full_stocks=[]

for root,dirs,files in os.walk(directory):
    for name in files:
#        csv_name = pd.DataFrame({'name' : name}, index = range(1))
        csv_list.append(os.path.join(root, name))
#        stock_csv = pd.read_csv(os.path.join(root, name))
#        leng = stock_csv.shape[0]
#        csv_name.append([csv_name]*leng, ignore_index=True)
#        stocks = stock_csv.join(csv_name)
#        full_stocks.concat(stocks)
       
#Combined all files into one master file
combined_csv = pd.concat([pd.read_csv(f) for f in csv_list], axis=0)
