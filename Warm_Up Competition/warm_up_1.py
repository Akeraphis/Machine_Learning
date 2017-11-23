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

#Plotting the data
f, a = plt.subplots(1, 3, sharey=True, tight_layout=True)

#Number of months since last donation
plt.subplot(3,1,1)
plt.hist(training[:,1]);
plt.xlabel('Months since last Donation');
plt.ylabel('Number of participants');
plt.title('Months since last donation');

#Number of Donations
plt.subplot(3,1,2)
plt.hist(training[:,2]);
plt.xlabel('Number of Donations');
plt.ylabel('Number of participants');
plt.title('Number of Donations');

#Number of Months x Number of Donations scatter
plt.subplot(3,1,3)
plt.plot(training[:,1], training[:,2], 'ro');
plt.xlabel('Monts since last donations');
plt.ylabel('Number of Donations');
plt.title('Number of Months since last donation by Number of Donations');

