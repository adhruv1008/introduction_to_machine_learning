#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y=train_test_split(features,labels,test_size=0.3,random_state=42)

clf=DecisionTreeClassifier()
clf.fit(train_X,train_y)
pred=clf.predict(test_X)
count=0
for i in range(len(pred)):
    if pred[i]==1 and test_y[i]==1:
        count+=1
### your code goes here 
print count

from sklearn.metrics import recall_score
print recall_score(pred,test_y)
