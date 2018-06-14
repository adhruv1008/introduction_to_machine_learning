#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

count=0
for i in enron_data.keys():
    if enron_data[i]['total_payments']=='NaN':
        count+=1
#print count

data=open("../final_project/poi_names.txt",'r')
#print data.read()
from feature_format import featureFormat

print featureFormat(enron_data,features,remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False)
#print enron_data
#print enron_data["SKILLING JEFFREY K"]['exercised_stock_options']