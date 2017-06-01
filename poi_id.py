#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot

###############################################################################
# PART 1: EXPLORE DATA SET
###############################################################################

### Load the dictionary containing the dataset 
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### number of samples
print "num of samples:", len(data_dict)
# there are 146 samples

### find number of features 
data_dict.keys()
type(data_dict['METTS MARK'])
print "features:", data_dict['METTS MARK'].keys()
print "num of features:", len(data_dict['METTS MARK'].keys())
# there are 21 features ("salary", "poi", etc.)

### find number of POI(person of interest) in data set
num_poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == 1:
        num_poi += 1
print "num poi:", num_poi
# there are 18 poi's

### find features with missing values
# 1.create function for finding NaN
def find_NaN(feature):
    num_NaN = 0
    num_NaN_poi = 0
    for person in data_dict:
        if data_dict[person][feature] == 'NaN':
            num_NaN += 1
            if data_dict[person]['poi'] == 1:
                num_NaN_poi += 1
    return [num_NaN, "%.3f" % (num_NaN/146.), num_NaN_poi, "%.3f" % (num_NaN_poi/18.)]

# 2. apply find_NaN() to all features 
NaN_dict = {}
for feature in data_dict['METTS MARK'].keys():
    NaN_dict[feature] = find_NaN(feature)
print "NaN dict:", NaN_dict 
# NaN_dict = {feature: [num_NaN, percent NaN, num_NaN_poi, percent poi NaN]}
# many features have missing values (NaN); poi's have less NaN than average

###############################################################################
# PART 2: OUTLIER DETECTION AND REMOVAL
###############################################################################

### set features to explore
features = ["salary", "bonus", "total_stock_value", "expenses", \
            "to_messages", "from_poi_to_this_person"]
data = featureFormat(data_dict, features)

### explore outliers in salary and bonus
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

for person in data_dict:
    if data_dict[person]["salary"] > 1000000 \
    and data_dict[person]["bonus"] > 5000000 \
    and data_dict[person]["salary"] != 'NaN' \
    and data_dict[person]["bonus"] != 'NaN':
        print person
# of the top 3 outliers, TOTAL is a dataset mistake and will be removed
# LAY KENNETH L and SKILLING JEFFREY K are both poi's with telling outliers

### remove outlier ("TOTAL")
data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features)

### explore outliers in total stock value and expenses
for point in data:
    total_stock_value = point[2]
    expenses = point[3]
    matplotlib.pyplot.scatter( total_stock_value , expenses )

matplotlib.pyplot.xlabel("total_stock_value")
matplotlib.pyplot.ylabel("expenses")
matplotlib.pyplot.show()

for person in data_dict:
    if data_dict[person]["total_stock_value"] > 3e7 \
    and data_dict[person]["total_stock_value"] != 'NaN':
        print "total_stock_value outlier:", person
    if data_dict[person]["expenses"] > 150000 \
    and data_dict[person]["expenses"] != 'NaN':
        print "expenses outlier:", person
# total_stock_value outliers: LAY KENNETH L and HIRKO JOSEPH; both poi's
# expenses outliers: SHANKMAN JEFFREY A, URQUHART JOHN A, MCCLELLAN GEORGE; none are poi's
# the outliers in the features do not seem to be errors, and will not be removed       

### explore outliers in to_messages and from_poi_to_this_person
for point in data:
    to_messages = point[4]
    from_poi_to_this_person = point[5]
    matplotlib.pyplot.scatter( to_messages , from_poi_to_this_person )

matplotlib.pyplot.xlabel("to_messages")
matplotlib.pyplot.ylabel("from_poi_to_this_person")
matplotlib.pyplot.show()

for person in data_dict:
    if data_dict[person]["to_messages"] > 10000 \
    and data_dict[person]["to_messages"] != 'NaN':
        print "to_messages outlier:", person
    if data_dict[person]["from_poi_to_this_person"] > 400 \
    and data_dict[person]["from_poi_to_this_person"] != 'NaN':
        print "from_poi_to_this_person outlier:", person
# to_messages outliers: SHAPIRO RICHARD S and KEAN STEVEN J; neither are poi's
# from_poi_to_this_person outlier: LAVORATO JOHN J; not poi
# the outliers in the features do not seem to be errors, and will not be removed

###############################################################################
# PART 3: CREATE NEW FEATURE
###############################################################################        

### create feature for number of NaN's
for person in data_dict:
    NaN_num = 0
    for feature in data_dict[person]:
        if data_dict[person][feature] == 'NaN':
            NaN_num += 1
    data_dict[person]["NaN_feature"] = NaN_num

# test new feature
data_dict['METTS MARK']['NaN_feature']
# METTS MARK has 'NaN' for 7 of his features

### Store to my_dataset for easy export below.
my_dataset = data_dict

###############################################################################
# PART 4: SET-UP ROUGH DRAFT OF CLASSIFIER
###############################################################################

# create features_list (start with all features)
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', \
'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', \
'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', \
'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', \
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person', 'NaN_feature']

# extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# split data into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

# initialize classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# I'll start with RandomForest since it has less bias and usually gets decent results

###############################################################################
# PART 5: VALIDATION
###############################################################################

# test accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print "RandomForest metrics result:"
print "accuracy:", accuracy

# test precision and recall scores
from sklearn import metrics
precision_score = metrics.precision_score(labels_test, pred)
recall_score = metrics.recall_score(labels_test, pred)
print "precision score:", precision_score
print "recall score:", recall_score

#==============================================================================
# RandomForest metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.333333333333
#==============================================================================

# since I'll have to re-test the metrics often, I'll create a function for it
def validation_scores(labels_test, pred):
    accuracy = accuracy_score(labels_test, pred)
    precision_score = metrics.precision_score(labels_test, pred)
    recall_score = metrics.recall_score(labels_test, pred)
    print "accuracy:", accuracy
    print "precision score:", precision_score
    print "recall score:", recall_score

###############################################################################
# PART 6: COMPARE ALGORITHMS
###############################################################################

# I'll try other algorithms and see how they compare 
# to my initial RandomForest classifer.

# try GaussianNB classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "GaussianNB metrics result:"
validation_scores(labels_test, pred)

#==============================================================================
# GaussianNB metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
#==============================================================================

# try DecisionTree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "DecisionTree metrics result:"
validation_scores(labels_test, pred)

#==============================================================================
# DecisionTree metrics result:
# accuracy: 0.827586206897
# precision score: 0.333333333333
# recall score: 0.666666666667
#==============================================================================

# From the metrics results, RandomForest classifier had the best results,
# so I will continue with it. 

###############################################################################
# PART 6: FEATURE SCALING USING MINMAXSCALER
###############################################################################

# Since features such as 'bonuses' can range in the millions and features such 
# as 'to_messages' can be much lower, I'll try using a feature scaler 

# scale data using MinMaxScaler()
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# re-initialize classifier with rescaled features
clf = RandomForestClassifier()
clf = clf.fit(features_train_scaled, labels_train)
pred = clf.predict(features_test_scaled)

# re-test metrics
print "MinMax scaling metrics result:"
validation_scores(labels_test, pred)

#==============================================================================
# MinMax scaling metrics result:
# accuracy: 0.862068965517
# precision score: 0.333333333333
# recall score: 0.333333333333
#==============================================================================

# the MinMaxScaler seems to have made the precision drop a bit, 
# perhaps an important feature has been scaled
# I'll use feature selection to remove noise

###############################################################################
# PART 7: FEATURE SELECTION USING SELECTKBEST
###############################################################################

# use selectKBest to see scores of all features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k='all')
selector.fit(features_train_scaled, labels_train)
print selector.scores_ 
# There are 13 features with an F-value above the F critical (3.91)

# keep 13 features with F-value above F critical
selector = SelectKBest(f_classif, k=13)
selector.fit(features_train_scaled, labels_train)
features_train_KBest = selector.transform(features_train_scaled)
features_test_KBest = selector.transform(features_test_scaled)

# re-initialize classifier with selectKbest features
clf = RandomForestClassifier()
clf = clf.fit(features_train_KBest, labels_train)
pred = clf.predict(features_test_KBest)

# re-test metrics
print "SelectKBest metrics result:"
validation_scores(labels_test, pred)

#==============================================================================
# SelectKBest metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.333333333333
#==============================================================================

# Slight increase in precision score

###############################################################################
# PART 8: FEATURE SELECTION USING PCA
###############################################################################

# try PCA for feature selection
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=12, whiten=True).fit(features_train_KBest) 
pca.explained_variance_ratio_ 
# the last two components have much less variance and will be removed

# transform with PCA
pca = RandomizedPCA(n_components=10, whiten=True).fit(features_train_KBest)
features_train_pca = pca.transform(features_train_KBest)
features_test_pca = pca.transform(features_test_KBest)

# re-initialize classifer with pca
clf = RandomForestClassifier()
clf = clf.fit(features_train_pca, labels_train)
pred = clf.predict(features_test_pca)

# re-test metrics
print "PCA metrics result:"
validation_scores(labels_test, pred)

#==============================================================================
# PCA metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.333333333333
#==============================================================================

#==============================================================================
# ###############################################################################
# # PART 8: TUNE PARAMETERS USING GRIDSEARCHCV
# ###############################################################################
# 
# from sklearn.grid_search import GridSearchCV
# from time import time
# 
# print "Fitting the classifier to the training set"
# t0 = time()
#          
# param_grid = {"n_estimators": [2, 5, 10, 50, 100],
#               "min_samples_split": [2, 5, 10, 50],
#               "oob_score": [False, True]}             
# 
# clf = RandomForestClassifier()
# grid_search = GridSearchCV(clf, param_grid)
# clf = grid_search.fit(features_train_pca, labels_train)
# pred = clf.predict(features_test_pca)
# 
# print "done in %0.3fs" % (time() - t0)
# print "Best estimator found by grid search:"
# print grid_search.best_estimator_
# # GridSearch best param: n_estimators=2, min_samples_split=2, oob_score=False
# 
# print "GridSearch metrics result:"
# validation_scores(labels_test, pred)
# 
# #==============================================================================
# # GridSearch metrics result:
# # accuracy: 0.896551724138
# # precision score: 0.5
# # recall score: 0.333333333333
# #==============================================================================
# 
# # GridSearch was scoring by accuracy by default, I'll change it to F1
# print "Fitting the classifier to the training set"
# t0 = time()
# 
# clf = RandomForestClassifier()
# grid_search = GridSearchCV(clf, param_grid, scoring='f1')
# clf = grid_search.fit(features_train_pca, labels_train)
# pred = clf.predict(features_test_pca)
# 
# print "done in %0.3fs" % (time() - t0)
# print "Best estimator found by grid search:"
# print grid_search.best_estimator_
# 
# print "GridSearch metrics result:"
# validation_scores(labels_test, pred)
#==============================================================================

# GridSearch best param: n_estimators=5, min_samples_split=2, oob_score=True
#==============================================================================
# GridSearch metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.333333333333
#==============================================================================

# Results of running the GridSearch a few times: 

# GridSearch best param: n_estimators=2, min_samples_split=5, oob_score=True
#==============================================================================
# GridSearch metrics result:
# accuracy: 0.896551724138
# precision score: 0.0
# recall score: 0.0
#==============================================================================

# GridSearch best param: n_estimators=50, min_samples_split=10, oob_score=False
#==============================================================================
# GridSearch metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.333333333333
#==============================================================================

# The GridSearch best param keeps changing, perhaps the sample data is too small,
# I'll try manually tuning the parameters

###############################################################################
# PART 8: TUNE PARAMETERS MANUALLY
###############################################################################

# I'll use the following outline for tuning parameters
clf = RandomForestClassifier()
clf = clf.fit(features_train_pca, labels_train)
pred = clf.predict(features_test_pca)
print "RF param tuning metrics result:"
validation_scores(labels_test, pred)

# best params:oob_score=True, n_estimators=100, min_samples_split=2

# for the following results I changed the test_size from .2 to .3

#==============================================================================
# oob_score=False metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
#==============================================================================

#==============================================================================
# oob_score=True metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
#==============================================================================
# similar metrics for oob_score=False and oob_score=True; 
# I'll use oob_score=True since the tester data has a large sample

#==============================================================================
# n_estimators=2 metrics result:
# accuracy: 0.863636363636
# precision score: 0.25
# recall score: 0.25
#==============================================================================

#==============================================================================
# n_estimators=5 metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
#==============================================================================

#==============================================================================
# n_estimators=10 metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
#==============================================================================

#==============================================================================
# n_estimators=50 metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
#==============================================================================

#==============================================================================
# n_estimators=100 metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
#==============================================================================
# since n_estimators= 5, 10, 50, and 100 are similar, 
# I'll go with 100 because the tester sample is larger

#==============================================================================
# min_samples_split=2 metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
#==============================================================================

#==============================================================================
# min_samples_split=5 metrics result:
# accuracy: 0.886363636364
# precision score: 0.333333333333
# recall score: 0.25
#==============================================================================

#==============================================================================
# min_samples_split=10 metrics result:
# accuracy: 0.886363636364
# precision score: 0.333333333333
# recall score: 0.25
#==============================================================================

#==============================================================================
# min_samples_split=20 metrics result:
# accuracy: 0.886363636364
# precision score: 0.333333333333
# recall score: 0.25
#==============================================================================
# min_samples_split=2 has best metrics

###############################################################################
# PART 8: COMPARE TUNED ALGORITHMS
###############################################################################

# tune GaussianNB classifier
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.metrics import f1_score

# non-poi:poi breakdown by training set and tester.py set
priors_training = [0.8767, 0.1233]
priors_tester = [0.8667, 0.1333]   

print "Fitting the clf_NB to the training set"
t0 = time()
      
param_grid_NB = {"priors": [None, priors_training, priors_tester]}             

clf_NB = GaussianNB()
grid_search_NB = GridSearchCV(clf_NB, param_grid_NB, scoring='f1')
clf_NB = grid_search_NB.fit(features_train_pca, labels_train)
pred = clf_NB.predict(features_test_pca)

print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print grid_search_NB.best_estimator_

print "clf_NB GridSearch metrics result:"
validation_scores(labels_test, pred)
print "F1 score:", f1_score(labels_test, pred)

#==============================================================================
# GaussianNB(priors=None)
# clf_NB GridSearch metrics result:
# accuracy: 0.840909090909
# precision score: 0.285714285714
# recall score: 0.5
# F1 score: 0.363636363636
#==============================================================================

# tune DecisionTree classifier
print "Fitting the clf_NB to the training set"
t0 = time()
      
param_grid_DT = {"min_samples_split": [2, 5, 10, 20, 50]}             

clf_DT = tree.DecisionTreeClassifier()
grid_search_DT = GridSearchCV(clf_DT, param_grid_DT, scoring='f1')
clf_DT = grid_search_DT.fit(features_train_pca, labels_train)
pred = clf_DT.predict(features_test_pca)

print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print grid_search_DT.best_estimator_

print "clf_DT GridSearch metrics result:"
validation_scores(labels_test, pred)
print "F1 score:", f1_score(labels_test, pred)

#==============================================================================
# DecisionTreeClassifier(min_samples_split=10)
# clf_DT GridSearch metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
# F1 score: 0.333333333333
#==============================================================================

# compare to RandomForest with F1 score
clf = RandomForestClassifier(oob_score=True, n_estimators=100, min_samples_split=2)
clf = clf.fit(features_train_pca, labels_train)
pred = clf.predict(features_test_pca)

print "clf_RF tuned metrics result:"
validation_scores(labels_test, pred)
print "F1 score:", f1_score(labels_test, pred)

#==============================================================================
# clf_RF tuned metrics result:
# accuracy: 0.909090909091
# precision score: 0.5
# recall score: 0.25
# F1 score: 0.333333333333
#==============================================================================

# After feature scaling, feature selection, and parameter tuning, 
# GaussianNB resulted with the highest metrics and will be the finalized classifier

###############################################################################
# PART 8: FINALIZE ALGORITHM
###############################################################################

### re-test SelectKBest on GaussianNB classifier 
###############################################################################
import matplotlib.pyplot as plt
from collections import OrderedDict
import operator

precision = []
recall = []
F1 = []
for n in range(1, len(features_list)):
    selector = SelectKBest(f_classif, k=n)
    selector.fit(features_train_scaled, labels_train)
    features_train_KBest = selector.transform(features_train_scaled)
    features_test_KBest = selector.transform(features_test_scaled)
    
    clf_NB = GaussianNB()
    clf_NB = clf_NB.fit(features_train_KBest, labels_train)
    pred = clf_NB.predict(features_test_KBest)

    precision_score = metrics.precision_score(labels_test, pred)
    recall_score = metrics.recall_score(labels_test, pred)
    F1_score = f1_score(labels_test, pred)
    
    precision.append(precision_score)
    recall.append(recall_score)
    F1.append(F1_score)

fig, ax = plt.subplots()    
plt.plot(precision)
plt.plot(recall)
plt.plot(F1)
plt.legend(['precision','recall', 'F1'])
ax.set_xlabel('Number of Features(k)')
ax.set_ylabel('Metric Scores')
plt.show()

# from the plot, k=1 has the best metrics, 
# it's likely that this is biased by the limited data, I'll test it out anyways

# apply SelectKBest(k=1) on GaussianNB
selector = SelectKBest(f_classif, k=1)
selector.fit(features_train_scaled, labels_train)
features_train_KBest = selector.transform(features_train_scaled)
features_test_KBest = selector.transform(features_test_scaled)

clf_NB = GaussianNB()
clf_NB = clf_NB.fit(features_train_KBest, labels_train)
pred = clf_NB.predict(features_test_KBest)

# running k=1 resulted in severely decreased metrics
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.862068965517
# precision score: 0.333333333333
# recall score: 0.333333333333
# F1 score: 0.333333333333
# Tester.py metrics result:        
# Accuracy: 0.84380       Precision: 0.33170      Recall: 0.16900 F1: 0.22392 
#==============================================================================

# try another method and select number of features based on feature scores
feature_scores = [11.77626989, 3.42318802, 0.04629757, 8.08602509, 18.06326173,
                  23.82368507, 6.78469145, 7.99107391, 0.7502535, 17.31847322,
                  3.02774957, 6.49347553, 0.4434606, 3.80331747, 0.09886609,
                  1.70185547, 7.51318039, 5.87227328, 4.27879888, 7.80551704]

# line plot of feature scores
plt.plot(feature_scores)
plt.show()

# bar plot of features and feature scores
feat_name_score_dict = OrderedDict()
for i in range(len(feature_scores)):
    try:
        feat_name_score_dict[features_list[i+1]] = feature_scores[i]
    except:
        pass
    
sorted_names_scores = sorted(feat_name_score_dict.items(), key=operator.itemgetter(1), reverse=True)
    
feature_scores.sort(reverse=True)
print feature_scores

x= list(range(len(sorted_names_scores)))
y = [i[1] for i in sorted_names_scores]
fig, ax = plt.subplots()
width = 0.3
rects = plt.bar(x,y,width)

ax.set_xticks(x)
ax.set_xticklabels([i[0] for i in sorted_names_scores], rotation='vertical')
ax.set_xlabel('Feature')
ax.set_ylabel('Feature Score')
plt.show()

# the plots show that starting from the 5th feature, the score plateaus;
# therefore I'll be using the top 4 features

# apply SelectKBest(k=4) on GaussianNB
selector = SelectKBest(f_classif, k=20)
selector.fit(features_train_scaled, labels_train)
features_train_KBest = selector.transform(features_train_scaled)
features_test_KBest = selector.transform(features_test_scaled)

clf_NB = GaussianNB()
clf_NB = clf_NB.fit(features_train_KBest, labels_train)
pred = clf_NB.predict(features_test_KBest)

# k=4; the results are still lower than required
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.931034482759
# precision score: 0.666666666667
# recall score: 0.666666666667
# F1 score: 0.666666666667
# Tester.py metrics results:
# Accuracy: 0.83747       Precision: 0.34290      Recall: 0.23900 F1: 0.28167     F2: 0.25442
#==============================================================================

# I'll try the remainder of the the possible k values
# k=2 
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.666666666667
# F1 score: 0.571428571429
# Tester.py metrics results:
# Accuracy: 0.84287       Precision: 0.34218      Recall: 0.19350 F1: 0.24721     F2: 0.21192        
#==============================================================================

# k=3
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.931034482759
# precision score: 0.666666666667
# recall score: 0.666666666667
# F1 score: 0.666666666667
# Tester.py metrics results:
# Accuracy: 0.84693       Precision: 0.37646      Recall: 0.22550 F1: 0.28205     F2: 0.24516 
#==============================================================================

# k=5
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.666666666667
# F1 score: 0.571428571429
# Tester.py metrics results:
# Accuracy: 0.83827       Precision: 0.35331      Recall: 0.25650 F1: 0.29722     F2: 0.27137
#==============================================================================

# k=6
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.666666666667
# F1 score: 0.571428571429
# Tester.py metrics results:
# Accuracy: 0.83607       Precision: 0.34649      Recall: 0.25900 F1: 0.29642     F2: 0.27278
#==============================================================================

# k=7
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.793103448276
# precision score: 0.2
# recall score: 0.333333333333
# F1 score: 0.25
# Tester.py metrics results:
# Accuracy: 0.83373       Precision: 0.34696      Recall: 0.28000 F1: 0.30991     F2: 0.29124
#==============================================================================

# k=8
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.793103448276
# precision score: 0.2
# recall score: 0.333333333333
# F1 score: 0.25
# Tester.py metrics results:
# Accuracy: 0.82747       Precision: 0.32847      Recall: 0.28150 F1: 0.30318     F2: 0.28979
#==============================================================================

# k=9
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.793103448276
# precision score: 0.2
# recall score: 0.333333333333
# F1 score: 0.25
# Tester.py metrics results:
# Accuracy: 0.82693       Precision: 0.33892      Recall: 0.31350 F1: 0.32571     F2: 0.31827    
#==============================================================================

# k=10
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results:
# Accuracy: 0.82520       Precision: 0.33528      Recall: 0.31650 F1: 0.32562     F2: 0.32008
#==============================================================================

# k=11
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results:
# Accuracy: 0.82387       Precision: 0.33087      Recall: 0.31400 F1: 0.32222     F2: 0.31724
#==============================================================================

# k=12
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.862068965517
# precision score: 0.333333333333
# recall score: 0.333333333333
# F1 score: 0.333333333333
# Tester.py metrics results:
# Accuracy: 0.82647       Precision: 0.34625      Recall: 0.33950 F1: 0.34284     F2: 0.34083
#==============================================================================

# k=13
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.862068965517
# precision score: 0.333333333333
# recall score: 0.333333333333
# F1 score: 0.333333333333
# Tester.py metrics results:
# Accuracy: 0.83393       Precision: 0.37113      Recall: 0.35350 F1: 0.36210     F2: 0.35689
#==============================================================================

# k=14; highest tester.py metrics!!!
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results:
# Accuracy: 0.83873       Precision: 0.39003      Recall: 0.37150 F1: 0.38054     F2: 0.37506    
#==============================================================================

# k=15    
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714    
# Tester.py metrics results:
# Accuracy: 0.83293       Precision: 0.36986      Recall: 0.35950 F1: 0.36460     F2: 0.36152
#==============================================================================

# k=16
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.793103448276
# precision score: 0.2
# recall score: 0.333333333333
# F1 score: 0.25    
# Tester.py metrics results:    
# Accuracy: 0.83447       Precision: 0.37161      Recall: 0.34950 F1: 0.36022     F2: 0.35371    
#==============================================================================

# k=17
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.793103448276
# precision score: 0.2
# recall score: 0.333333333333
# F1 score: 0.25
# Tester.py metrics results: 
# Accuracy: 0.83913       Precision: 0.38366      Recall: 0.34050 F1: 0.36079     F2: 0.34834
#==============================================================================

# k=18
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.793103448276
# precision score: 0.2
# recall score: 0.333333333333
# F1 score: 0.25
# Tester.py metrics results: 
# Accuracy: 0.84247       Precision: 0.39611      Recall: 0.34600 F1: 0.36936     F2: 0.35498
#==============================================================================

# k=19
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.333333333333
# recall score: 0.666666666667
# F1 score: 0.444444444444
# Tester.py metrics results:    
# Accuracy: 0.84227       Precision: 0.39788      Recall: 0.35650 F1: 0.37605     F2: 0.36407
#==============================================================================

# k=20
#==============================================================================
# F1 score: 0.5
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.333333333333
# recall score: 0.666666666667
# F1 score: 0.444444444444
# Tester.py metrics results: 
# Accuracy: 0.83767       Precision: 0.39406      Recall: 0.40450 F1: 0.39921     F2: 0.40237    
#==============================================================================
    
# k=14 has scores comparable to k=20, but less features needed

# create metrics lists    
KBest_tester_precision = [0.33170, 0.34218, 0.37646, 0.34290, 0.34290, 0.35331, \
0.34649, 0.32847, 0.33892, 0.33528, 0.33087, 0.34625, 0.37113, 0.39003, 0.36986, \
0.37161, 0.38366, 0.39611, 0.39788, 0.39406]
KBest_tester_recall = [0.16900, 0.19350, 0.22550, 0.23900, 0.23900, 0.25650, \
0.25900, 0.28150, 0.31350, 0.31650, 0.31400, 0.33950, 0.35350, 0.37150, 0.35950, \
0.34950, 0.34050, 0.34600, 0.35650, 0.40450]
KBest_tester_F1 = [0.22392, 0.24721, 0.28205, 0.28167, 0.28167, 0.29722, \
0.29642, 0.30318, 0.32571, 0.32562, 0.32222, 0.34284, 0.36210, 0.38054, 0.36460, \
0.36022, 0.36079, 0.36936, 0.37605, 0.39921]

# plot SelectKBest metrics
fig, ax = plt.subplots()    
plt.plot(KBest_tester_precision)
plt.plot(KBest_tester_recall)
plt.plot(KBest_tester_F1)
plt.legend(['precision','recall', 'F1'], loc='bottom right')
ax.set_xlabel('Number of Features(k)')
ax.set_ylabel('Tester.py Metric Scores')
plt.show()

# set k=14
selector = SelectKBest(f_classif, k=14)
selector.fit(features_train_scaled, labels_train)
features_train_KBest = selector.transform(features_train_scaled)
features_test_KBest = selector.transform(features_test_scaled)

clf_NB = GaussianNB()
clf_NB = clf_NB.fit(features_train_KBest, labels_train)
pred = clf_NB.predict(features_test_KBest)    

### re-test PCA on GaussianNB
###############################################################################

pca = RandomizedPCA(n_components=14, whiten=True).fit(features_train_KBest) 
print pca.explained_variance_ratio_ 

# plot variance ratio
fig, ax = plt.subplots()    
pca_variance = pca.explained_variance_ratio_
plt.plot(pca_variance)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Explained Variance Ratio')
plt.show()

# try different n_components; n_components=10 has highest metrics
pca = RandomizedPCA(n_components=10, whiten=True).fit(features_train_KBest)

# n_components=1
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.793103448276
# precision score: 0.0
# recall score: 0.0
# F1 score: 0.0
# Tester.py metrics results:
# Accuracy: 0.86580       Precision: 0.48858      Recall: 0.13900 F1: 0.21643     F2: 0.16221    
#==============================================================================
    
# n_components=2
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results:
# Accuracy: 0.86340       Precision: 0.47664      Recall: 0.25000 F1: 0.32798     F2: 0.27627    
#==============================================================================

# n_components=3
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results:
# Accuracy: 0.85293       Precision: 0.42209      Recall: 0.27900 F1: 0.33594     F2: 0.29929    
#==============================================================================

# n_components=4
#==============================================================================
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results: 
# Accuracy: 0.83840       Precision: 0.36269      Recall: 0.28000 F1: 0.31603     F2: 0.29338    
#==============================================================================

# n_components=5
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results: 
# Accuracy: 0.83907       Precision: 0.37866      Recall: 0.32300 F1: 0.34862     F2: 0.33278
#==============================================================================

# n_components=6
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results: 
# Accuracy: 0.84060       Precision: 0.38911      Recall: 0.34300 F1: 0.36460     F2: 0.35133
#==============================================================================
 
# n_components=7
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results: 
# Accuracy: 0.83900       Precision: 0.38323      Recall: 0.34050 F1: 0.36060     F2: 0.34827
#==============================================================================

# n_components=8
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results: 
# Accuracy: 0.84107       Precision: 0.39345      Recall: 0.35450 F1: 0.37296     F2: 0.36166
#==============================================================================

# n_components=9
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results: 
# Accuracy: 0.84187       Precision: 0.39825      Recall: 0.36400 F1: 0.38036     F2: 0.37037
#==============================================================================

# n_components=10; highest tester.py metrics!!!
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.827586206897
# precision score: 0.25
# recall score: 0.333333333333
# F1 score: 0.285714285714
# Tester.py metrics results:
# Accuracy: 0.83873       Precision: 0.39003      Recall: 0.37150 F1: 0.38054     F2: 0.37506
#==============================================================================

# n_components=11
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.333333333333
# F1 score: 0.4
# Tester.py metrics results: 
# Accuracy: 0.83160       Precision: 0.36993      Recall: 0.37400 F1: 0.37195     F2: 0.37318
#==============================================================================
 
# n_components=12
#==============================================================================
# Pipeline metrics result:
# accuracy: 0.896551724138
# precision score: 0.5
# recall score: 0.333333333333
# F1 score: 0.4
# Tester.py metrics results: 
# Accuracy: 0.82767       Precision: 0.36196      Recall: 0.38350 F1: 0.37242     F2: 0.37899
#==============================================================================

# create metrics list
PCA_tester_precision = [0.48858, 0.47664, 0.42209, 0.36269, 0.37866, 0.38911, \
0.38323, 0.39345, 0.39825, 0.39003, 0.36993, 0.36196]
PCA_tester_recall = [0.13900, 0.25000, 0.27900, 0.28000, 0.32300, 0.34300, \
0.34050, 0.35450, 0.36400, 0.37150, 0.37400, 0.38350]
PCA_tester_F1 = [0.21643, 0.32798, 0.33594, 0.31603, 0.34862, 0.36460, \
0.36060, 0.37296, 0.38036, 0.38054, 0.37195, 0.37242]

# plot SelectKBest metrics
fig, ax = plt.subplots()    
plt.plot(PCA_tester_precision)
plt.plot(PCA_tester_recall)
plt.plot(PCA_tester_F1)
plt.legend(['precision','recall', 'F1'], loc='bottom right')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Tester.py Metric Scores')
plt.show()

 
###############################################################################
# PART 8: CREATE PIPELINE 
###############################################################################

# create pipeline for easy transfer
from sklearn.pipeline import Pipeline
clf = Pipeline([('MinMaxScaler', scaler), ('SelectKBest', selector), ('pca', pca), ('GaussianNB', clf_NB)])

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print "Pipeline metrics result:"
validation_scores(labels_test, pred)
print "F1 score:", f1_score(labels_test, pred)



'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
'''
dump_classifier_and_data(clf, my_dataset, features_list)