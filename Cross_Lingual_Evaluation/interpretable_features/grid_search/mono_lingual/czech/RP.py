BASE_OUT_PATH = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
SVM_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters/CZECH/RP/SVM/RP.txt'
KNN_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters/CZECH/RP/KNN/RP.txt'
RF_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters/CZECH/RP/RF/RP.txt'
XG_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters/CZECH/RP/XG/RP.txt'
BAGG_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperpameters/CZECH/RP/BAGG/RP.txt'

import sys
sys.path.append("/export/b15/afavaro/git_code_version/speech")
from Cross_Lingual_Evaluation.interpretable_features.classification.mono_lingual.Data_Prep_RP import *
from Cross_Lingual_Evaluation.interpretable_features.classification.mono_lingual.Utils import *
import numpy as np
import random
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
random.seed(10)

czech_data = czech_prep(os.path.join(BASE, "Czech/final_data_experiments_updated.csv"))
gr = czech_data.groupby('labels')
ctrl_ = gr.get_group(0)
pd_ = gr.get_group(1)

arrayOfSpeaker_cn = ctrl_['names'].unique()
random.shuffle(arrayOfSpeaker_cn)

arrayOfSpeaker_pd = pd_['names'].unique()
random.shuffle(arrayOfSpeaker_pd)

cn_sps = get_n_folds(arrayOfSpeaker_cn)
pd_sps = get_n_folds(arrayOfSpeaker_pd)

data = []
for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
    data.append(cn_sp + pd_sp)
n_folds = sorted(data, key=len, reverse=True)

folds = []
for i in n_folds:
    data_i = czech_data[czech_data["names"].isin(i)]
    data_i = data_i.drop(columns=['names'])
    folds.append((data_i).to_numpy())


data_train_1, data_test_1, data_train_2, data_test_2, data_train_3, data_test_3, data_train_4, data_test_4, \
data_train_5, data_test_5,  data_train_6, data_test_6, data_train_7, data_test_7 , data_train_8, data_test_8, \
data_train_9, data_test_9, data_train_10, data_test_10 = create_split_train_test(folds)

#####################################################################################################################

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


    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = [1, 0.1, 0.01, 0.001]
    grid = dict(kernel=kernel, C=C, gamma=gamma)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    # summarize result
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
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
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
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
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, config in zip(means, params):
        config = str(config)
        if config in rf_paramters:
            rf_paramters[config].append(mean)
        else:
            rf_paramters[config] = [mean]

    # GradientBoostingClassifier
    model = GradientBoostingClassifier()
    n_estimators = [10, 100, 1000]
    learning_rate = [0.001, 0.01, 0.1]
    subsample = [0.5, 0.7, 1.0]
    max_depth = [3, 7, 9]
    # define grid search
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    means = grid_result.cv_results_['mean_test_score']
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
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train)
    # summarize results
   # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print(grid_result.best_params_)
  #  path = os.path.join(store_parameters, f"{i}.txt")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, config in zip(means, params):
        config = str(config)
        if config in bagg_paramters:
            bagg_paramters[config].append(mean)
        else:
            bagg_paramters[config] = [mean]

################################################################################################################

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

fo = open(SVM_OUT_PATH, "w")
for k, v in svm_parameters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(KNN_OUT_PATH, "w")
for k, v in knn_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(RF_OUT_PATH, "w")
for k, v in rf_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(XG_OUT_PATH, "w")
for k, v in xg_paramters.items():
    fo.write(str(k) + ' >>> '+ str(v) + '\n\n')

fo = open(BAGG_OUT_PATH, "w")
for k, v in bagg_paramters.items():
    fo.write(str(k) + ' >>> ' + str(v) + '\n\n')

