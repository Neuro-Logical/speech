#!/usr/bin/env python
# coding: utf-8
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as pl
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils import shuffle
import random
import os
from Utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils import shuffle
import random
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
from sklearn.inspection import plot_partial_dependence
random.seed(10)

spain = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv")
spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
spain['task'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
spain['id'] = [elem.split("_")[2].split("-")[0] for elem in spain['AudioFile'].tolist()]
task = ['concatenateread']
spain = spain[spain['task'].isin(task)]
spain=spain.drop(columns=['AudioFile', 'task', 'Unnamed: 0'])

lab = []
for m in spain['labels'].tolist():
    if m == 'PD':
        lab.append(1)
    if m == 'HC':
        lab.append(0)
spain['labels'] = lab

gr = spain.groupby('labels')
ctrl_ = gr.get_group(0)
pd_ = gr.get_group(1)
arrayOfSpeaker_cn = ctrl_['id'].unique()
random.shuffle(arrayOfSpeaker_cn)

arrayOfSpeaker_pd = pd_['id'].unique()
random.shuffle(arrayOfSpeaker_pd)

cn_sps = get_n_folds(arrayOfSpeaker_cn)
pd_sps = get_n_folds(arrayOfSpeaker_pd)

data = []
for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
    data.append(cn_sp + pd_sp)
n_folds = sorted(data, key=len, reverse=True)

folds = []
for i in n_folds:
    data_i = spain[spain["id"].isin(i)]
    data_i = data_i.drop(columns=['id'])
    folds.append((data_i).to_numpy())

data_train_1 = np.concatenate(folds[:9])
data_test_1 = np.concatenate(folds[-1:])

data_train_2 = np.concatenate(folds[1:])
data_test_2 = np.concatenate(folds[:1])

data_train_3 = np.concatenate(folds[2:] + folds[:1])
data_test_3 = np.concatenate(folds[1:2])

data_train_4 = np.concatenate(folds[3:] + folds[:2])
data_test_4 = np.concatenate(folds[2:3])

data_train_5 = np.concatenate(folds[4:] + folds[:3])
data_test_5 = np.concatenate(folds[3:4])

data_train_6 = np.concatenate(folds[5:] + folds[:4])
data_test_6 = np.concatenate(folds[4:5])

data_train_7 = np.concatenate(folds[6:] + folds[:5])
data_test_7 = np.concatenate(folds[5:6])

data_train_8 = np.concatenate(folds[7:] + folds[:6])
data_test_8 = np.concatenate(folds[6:7])

data_train_9 = np.concatenate(folds[8:] + folds[:7])
data_test_9 = np.concatenate(folds[7:8])

data_train_10 = np.concatenate(folds[9:] + folds[:8])
data_test_10 = np.concatenate(folds[8:9])


svm_parameters = {}

rf_paramters = {}

knn_paramters = {}

xg_paramters = {}

bagg_paramters = {}


for i in range(1, 11):

    print(i)

    normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"),
                                                                                       eval(f"data_test_{i}"))
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(normalized_train_X, y_train)
    model = SelectFromModel(clf, prefit=True, max_features=30)

    X_train = model.transform(normalized_train_X)
    cols = model.get_support(indices=True)

    X_test = normalized_test_X[:, cols]
    reduced_data = data_i.iloc[:, :-1]
    selected_features = reduced_data.columns[model.get_support()].to_list()

    # SVC
    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = [1, 0.1, 0.01, 0.001]
    # gamma = ['scale']
    # define grid search
    grid = dict(kernel=kernel, C=C, gamma=gamma)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    print(grid_result.best_params_)

    means = grid_result.cv_results_['mean_test_score']
    print(means)
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']


    for mean, config in zip(means, params):
        config = str(config)
        if config in svm_parameters:
            svm_parameters[config].append(mean)
        else:
            svm_parameters[config] = [mean]


    # KNeighborsClassifier
    model = KNeighborsClassifier()
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    # define grid search
    grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    # summarize results

    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    print(means)
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']


    for mean, config in zip(means, params):
        config = str(config)
        if config in knn_paramters:
            knn_paramters[config].append(mean)
        else:
            knn_paramters[config] = [mean]



    # RandomForestClassifier
    model = RandomForestClassifier()
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']
    # define grid search
    grid = dict(n_estimators=n_estimators, max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    print(grid_result.best_params_)


    means = grid_result.cv_results_['mean_test_score']
    print(means)
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']


    for mean, config in zip(means, params):
        config = str(config)
        if config in rf_paramters:
            rf_paramters[config].append(mean)
        else:
            rf_paramters[config] = [mean]

    # define dataset

    # define models and parameters

    model = GradientBoostingClassifier()
    n_estimators = [10, 100, 1000]
    learning_rate = [0.001, 0.01, 0.1]
    subsample = [0.5, 0.7, 1.0]
    max_depth = [3, 7, 9]
    # define grid search
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    # summarize results
  #  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    print(means)
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, config in zip(means, params):
        config = str(config)
        if config in xg_paramters:
            xg_paramters[config].append(mean)
        else:
            xg_paramters[config] = [mean]


    # BaggingClassifier

    model = BaggingClassifier()
    max_samples = [0.05, 0.1, 0.2, 0.5]
    n_estimators = [10, 100, 1000]
    # define grid search
    grid = dict(n_estimators=n_estimators, max_samples=max_samples)
    cv = RepeatedStratifiedKFold(n_splits=9, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    print(means)
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, config in zip(means, params):
        config = str(config)
        if config in bagg_paramters:
            bagg_paramters[config].append(mean)
        else:
            bagg_paramters[config] = [mean]





SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters_2/SPANISH/TDU/SVM/TDU.txt'

KNN = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters_2/SPANISH/TDU/KNN/TDU.txt'

RF = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters_2/SPANISH/TDU/RF/TDU.txt'

XG = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters_2/SPANISH/TDU/XG/TDU.txt'

BAGG = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters_2/SPANISH/TDU/BAGG/TDU.txt'




for k in svm_parameters.keys():
    svm_parameters[k] = np.array(svm_parameters[k]).mean()



for k in knn_paramters.keys():
    knn_paramters[k] = np.array(knn_paramters[k]).mean()


for k in rf_paramters.keys():
    rf_paramters[k] = np.array(rf_paramters[k]).mean()


for k in xg_paramters.keys():
    xg_paramters[k] = np.array(xg_paramters[k]).mean()


for k in bagg_paramters.keys():
    bagg_paramters[k] = np.array(bagg_paramters[k]).mean()




fo = open(SVM, "w")
for k, v in svm_parameters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')


fo = open(KNN, "w")
for k, v in knn_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')


fo = open(RF, "w")
for k, v in rf_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')


fo = open(XG, "w")
for k, v in xg_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(BAGG, "w")
for k, v in bagg_paramters.items():
    fo.write(str(k) + ' >>> ' + str(v) + '\n\n')

