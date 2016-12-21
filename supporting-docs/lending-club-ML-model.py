#!/usr/bin/python

from __future__ import division
import pandas as pd
import numpy as np


# Read the file into pandas
df = pd.read_csv("/LendingClubData/loans.csv")


# Load required sklearn modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics


# Set the prediction and target columns
predict_cols = ['int_rate','annual_inc','dti','acc_now_delinq','term','grade','last_fico_range_high','last_fico_range_low','num_tl_30dpd','percent_bc_gt_75','tot_cur_bal','tot_hi_cred_lim']
target_cols = ['default']


# Create X (predictors) and y (target) for data
X = df[predict_cols]
y = df.default


# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Make model pipeline and use cross-validated grid search to find best hyperparameters
pipe = make_pipeline(RandomForestClassifier(class_weight='balanced'))

param_grid = {'randomforestclassifier__n_estimators': [10, 50, 100, 150],
             'randomforestclassifier__min_samples_leaf': [5, 10, 20, 30, 50]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)


# Best parameters from grid search
grid.best_params_


# Score using a dummy classifier to compare
dummy = DummyClassifier()
dummy.fit(X_train, y_train)
dummy.score(X_test, y_test)


# Create Random Forest classifier using best hyperparameters
rfc = RandomForestClassifier(class_weight='balanced', min_samples_leaf=5, n_estimators=150)

rfc.fit(X_train, y_train)

# Create a predicted value variable for use in scoring
y_pred_class = rfc.predict(X_test)


# Print various scoring metrics
print "Confusion Matrix: \n %s" % metrics.confusion_matrix(y_test, y_pred_class)
print "Accuracy: %s" % metrics.accuracy_score(y_test, y_pred_class)
print "Error: %s" % (1 - metrics.accuracy_score(y_test, y_pred_class))
print "AUC: %s" % metrics.roc_auc_score(y_test, y_pred_class)
print "Precision: %s" % metrics.precision_score(y_test, y_pred_class)
print "Recall: %s" % metrics.recall_score(y_test, y_pred_class)

# List most important features to the classifier
featimp = pd.Series(rfc.feature_importances_, index=predict_cols).sort_values(ascending=False)
print "\nFeature Importance:"
print featimp

# Final fit of the model with all available data
rfc.fit(X,y)

# Pickle the model for future use.
from sklearn.externals import joblib

joblib.dump(rfc, "./lc-rfc-model.pkl", compress=3) 
