# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:08:42 2017

@author: jcd1
"""

from __future__ import division
from IPython.display import display
from matplotlib import pyplot as plt

import numpy  as np
import pandas as pd
import random, sys, os, re


from sklearn.linear_model     import LogisticRegression

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search      import RandomizedSearchCV, GridSearchCV
from sklearn.cross_validation import cross_val_predict, permutation_test_score

SEED   = 1091
scale  = False 
minmax = False
normd   = False
nointercept = True
engineering = True

N_CLASSES = 2

submission_filename = "../BloodDonationSubmissionFormat.csv"

from load_blood_data import load_blood_data

y_train, X_train = load_blood_data(train=True, SEED   = SEED, scale  = scale,minmax = minmax, norm   = normd,nointercept = nointercept, engineering = engineering)

StatifiedCV = StratifiedKFold(y            = y_train, n_folds      = 10,  shuffle      = True,  random_state = SEED)

random.seed(SEED)

clf = LogisticRegression(penalty           = 'l2',dual              = False,C                 = 1.0, fit_intercept     = True,solver            = 'liblinear',   max_iter          = 100, intercept_scaling = 1,tol               = 0.0001, class_weight      = None,random_state      = SEED, multi_class       = 'ovr',verbose           = 0,warm_start        = False,n_jobs            = -1)



# param_grid = dict(C             = [0.0001, 0.001, 0.01, 0.1],
#                   fit_intercept = [True, False],
#                   penalty       = ['l1', 'l2'],
#                   #solver        = ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
#                   max_iter      = [50, 100, 250])

# grid_clf = GridSearchCV(estimator  = clf, 
#                         param_grid = param_grid,
#                         n_jobs     = 1,  
#                         cv         = StatifiedCV).fit(X_train, y_train)

# print("clf_params = {}".format(grid_clf.best_params_))
# print("score: {}".format(grid_clf.best_score_))

# clf = grid_clf.best_estimator_




clf_params = {'penalty': 'l2', 'C': 0.001, 'max_iter': 50, 'fit_intercept': True}
clf.set_params(**clf_params)
clf.fit(X_train, y_train)
# from sklearn_utilities import GridSearchHeatmap

# GridSearchHeatmap(grid_clf, y_key='learning_rate', x_key='n_estimators')

# from sklearn_utilities import plot_validation_curves

# plot_validation_curves(grid_clf, param_grid, X_train, y_train, ylim = (0.0, 1.05))

train_preds = cross_val_predict(estimator    = clf, X            = X_train,y            = y_train,  cv           = StatifiedCV,  n_jobs       = -1, verbose      = 0,fit_params   = None, pre_dispatch = '2*n_jobs')

y_true, y_pred   = y_train, train_preds

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=None)

accuracy = round(np.trace(cm)/float(np.sum(cm)),4)
misclass = 1 - accuracy
print("Accuracy {}, mis-class rate {}".format(accuracy,misclass))

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=None)


plt.figure(figsize=(10,6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)

AUC = roc_auc_score(y_true, y_pred, average='macro')
plt.text(x=0.6,y=0.4,s="AUC         {:.4f}"\
         .format(AUC),
        fontsize=16)

plt.text(x=0.6,y=0.3,s="accuracy {:.2f}%"\
         .format(accuracy*100),
        fontsize=16)

logloss = log_loss(y_true, y_pred)
plt.text(x=0.6,y=0.2,s="LogLoss   {:.4f}"\
         .format(logloss),
        fontsize=16)

f1 = f1_score(y_true, y_pred)
plt.text(x=0.6,y=0.1,s="f1             {:.4f}"\
         .format(f1),
        fontsize=16)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

score, permutation_scores, pvalue = permutation_test_score(estimator      = clf,X              = X_train.values.astype(np.float32),y              = y_train,cv             = StatifiedCV, labels         = None,random_state   = SEED,verbose        = 0, n_permutations = 100, scoring        = None,n_jobs         = -1) 

plt.figure(figsize=(20,8))
plt.hist(permutation_scores, 20, label='Permutation scores')

ylim = plt.ylim()
plt.plot(2 * [score], ylim, '--g', linewidth=3,
         label='Classification Score (pvalue {:.4f})'.format(pvalue))
         
plt.plot(2 * [1. / N_CLASSES], ylim, 'r', linewidth=7, label='Luck')

plt.ylim(ylim)
plt.legend(loc='center',fontsize=16)
plt.xlabel('Score')
plt.show()

# find mean and stdev of the scores
from scipy.stats import norm
mu, std = norm.fit(permutation_scores)


clf.set_params(**clf_params)
clf.fit(X_train, y_train)
from load_blood_data import load_blood_data

X_test, IDs = load_blood_data(train=False, SEED   = SEED, scale  = scale, minmax = minmax,norm   = normd,nointercept = nointercept,engineering = engineering)

y_pred = clf.predict(X_test)
print(y_pred[:10])


y_pred_probs  = clf.predict_proba(X_test)
print(y_pred_probs[:10])
donate_probs  = [prob[1] for prob in y_pred_probs]

print(donate_probs[:10])

assert len(IDs)==len(donate_probs)

f = open(submission_filename, "w")

f.write(",Made Donation in March 2007\n")
for ID, prob in zip(IDs, donate_probs):
    f.write("{},{}\n".format(ID,prob))
    
f.close()