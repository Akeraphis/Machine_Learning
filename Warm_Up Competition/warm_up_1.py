# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:33:55 2017

@author: jcd1
"""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Import the datasets
full_training_dt = genfromtxt('training-data.csv', delimiter=',', skip_header=1)
testing_dt = genfromtxt('testing-data.csv', delimiter=',', skip_header=1)

#Shuffle the dataset and split into training, cv and testing
np.random.shuffle(full_training_dt)
training, test = np.split(full_training_dt, [462]);

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

plt.plot(training[:,3],training[:,2], 'ro');
plt.title('Total volume donated by Number of Donations');

plt.hist(training[:,4]);
plt.title('Monts since first donation');

plt.plot(training[:,4],training[:,1], 'ro');
plt.title('#months since last donation by #months since first donation');

sb.countplot(training[:,5], palette='hls');

#checking for missing values
np.isnan(training)

#checking for dependencies between values
sb.heatmap(np.corrcoef(training, rowvar=False))

#Logistic regression
LogReg = LogisticRegression()
LogReg.fit(training[:,1:4], training[:,5])
y_pred = LogReg.predict(test[:,1:4])
print(classification_report(test[:,5], y_pred))
