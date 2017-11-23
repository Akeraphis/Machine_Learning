# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:33:55 2017

@author: jcd1
"""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

#Import the datasets
full_training_dt = genfromtxt('training-data.csv', delimiter=',', skip_header=1)
testing_dt = genfromtxt('testing-data.csv', delimiter=',', skip_header=1)

#Shuffle the dataset and split into training, cv and testing
np.random.shuffle(full_training_dt)
training, cv, test = np.split(full_training_dt, [346, 462]);

