#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###from sklearn.metrics import accuracy_score

from sklearn import svm
clf=svm.SVC(kernel="rbf",C=10000)
t0=time()
clf.fit(features_train,labels_train)
print round(time()-t0,3)
pred=clf.predict(features_test)
import numpy as np
y=np.array(pred)
print np.count_nonzero(y==1)

#########################################################


