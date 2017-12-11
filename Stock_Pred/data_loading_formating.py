# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:25:50 2017

@author: JCD1
"""
#Load libraries
import numpy as np
from numpy import genfromtxt
from numpy import savetxt

#Load data files
A_quarterly_financial_data = genfromtxt(r'C:\Users\jcd1\Documents\GitHub\Machine_Learning\Stock_Pred\data\A_quarterly_financial_data.csv', delimiter=',', skip_header=1)
AA_quarterly_financial_data = genfromtxt(r'C:\Users\jcd1\Documents\GitHub\Machine_Learning\Stock_Pred\data\AA_quarterly_financial_data.csv', delimiter=',', skip_header=1)
