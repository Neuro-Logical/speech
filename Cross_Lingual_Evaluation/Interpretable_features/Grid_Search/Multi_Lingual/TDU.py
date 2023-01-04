BASE = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperparameters_Multi/GITA/TDU/SVM/SS.txt'
KNN = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperparameters_Multi/GITA/TDU/KNN/SS.txt'
RF = '//export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperparameters_Multi/GITA/TDU/RF/SS.txt'
XG = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperparameters_Multi/GITA/TDU/XG/SS.txt'
BAGG = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Best_hyperparameters_Multi/GITA/TDU/BAGG/SS.txt'

from Cross_Lingual_Evaluation.Interpretable_features.Classification.Multi_Lingual.Data_Prep_TDU import *
from Cross_Lingual_Evaluation.Interpretable_features.Classification.Multi_Lingual.Utils_TDU import *
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
random.seed(10)

colombian, colombian_cols = gita_prep(os.path.join(BASE, "/GITA/total_data_frame_novel_task_combined_ling_tot.csv"))
german, german_cols = german_prep(os.path.join(BASE, "/GERMAN/final_data_frame_with_intensity.csv"))
italian, italian_cols = italian_prep(os.path.join(BASE, "/ITALIAN_PD/tot_experiments_ling_fin.csv"))
spain, spain_cols = neurovoz_prep(os.path.join(BASE, "/NEUROVOZ/tot_data_experiments.csv"))

one_inter = IntersecOftwo(german_cols, italian_cols)
lista_to_keep = IntersecOfSets(one_inter, colombian_cols,spain_cols)

colombian = colombian[colombian.columns.intersection(lista_to_keep)]
german = german[german.columns.intersection(lista_to_keep)]
italian = italian[italian.columns.intersection(lista_to_keep)]
spain = spain[spain.columns.intersection(lista_to_keep)]

spain = preprocess_data_frame(spain)
spain_folds = create_n_folds(spain)

data_train_1_spain, data_test_1_spain, data_train_2_spain, data_test_2_spain, data_train_3_spain, data_test_3_spain, \
data_train_4_spain, data_test_4_spain, data_train_5_spain, data_test_5_spain, data_train_6_spain, data_test_6_spain, \
data_train_7_spain, data_test_7_spain, data_train_8_spain, data_test_8_spain, data_train_9_spain, data_test_9_spain, \
data_train_10_spain, data_test_10_spain = create_split_train_test(spain_folds)

german = preprocess_data_frame(german)
german_folds = create_n_folds(german)

data_train_1_german, data_test_1_german, data_train_2_german, data_test_2_german, \
data_train_3_german, data_test_3_german, data_train_4_german, data_test_4_german , \
data_train_5_german, data_test_5_german, data_train_6_german, data_test_6_german, \
data_train_7_german, data_test_7_german, data_train_8_german, data_test_8_german, \
data_train_9_german, data_test_9_german, data_train_10_german, \
data_test_10_german = create_split_train_test(german_folds)

colombian = preprocess_data_frame(colombian)
colombian_folds = create_n_folds(colombian)

data_train_1_colombian, data_test_1_colombian, data_train_2_colombian, \
data_test_2_colombian, data_train_3_colombian, data_test_3_colombian , \
data_train_4_colombian, data_test_4_colombian , data_train_5_colombian, \
data_test_5_colombian, data_train_6_colombian, data_test_6_colombian, \
data_train_7_colombian, data_test_7_colombian, data_train_8_colombian, \
data_test_8_colombian, data_train_9_colombian, data_test_9_colombian, \
data_train_10_colombian, data_test_10_colombian = create_split_train_test(colombian_folds)

# italian
italian = preprocess_data_frame(italian)
italian_folds = create_n_folds(italian)

data_train_1_italian, data_test_1_italian, data_train_2_italian, data_test_2_italian, \
data_train_3_italian, data_test_3_italian, data_train_4_italian, data_test_4_italian, data_train_5_italian,\
data_test_5_italian, data_train_6_italian, data_test_6_italian,  data_train_7_italian, data_test_7_italian, \
data_train_8_italian, data_test_8_italian, data_train_9_italian, data_test_9_italian, data_train_10_italian, \
data_test_10_italian = create_split_train_test(italian_folds)

#############################################################################################################################

svm_parameters = {}
rf_paramters = {}
knn_paramters = {}
xg_paramters = {}
bagg_paramters = {}

for i in range(1, 11):

    print(i)

    normalized_train_X_spain, normalized_test_X_spain, y_train_spain, y_test_spain = normalize(eval(f"data_train_{i}_spain"), eval(f"data_test_{i}_spain"))
    normalized_train_X_italian, normalized_test_X_italian, y_train_italian, y_test_italian = normalize(eval(f"data_train_{i}_italian"), eval(f"data_test_{i}_italian"))
    normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))

    normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

    training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_spain, y_train_spain, normalized_train_X_german,
                                                  y_train_german, normalized_train_X_italian, y_train_italian)


    test_data, test_labels = normalized_test_X_colombian, y_test_colombian

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(training_data, training_labels)
    model = SelectFromModel(clf, prefit=True, max_features=30)
    X_train = model.transform(training_data)
    cols = model.get_support(indices=True)
    X_test = test_data[:, cols]

    model = SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [50, 10, 1.0, 0.1, 0.01]
    gamma = [1, 0.1, 0.01, 0.001]
    # gamma = ['scale']
    # define grid search
    grid = dict(kernel=kernel, C=C, gamma=gamma)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_train, training_labels)
    print(grid_result.best_params_)

    means = grid_result.cv_results_['mean_test_score']
    print(max(means))
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']


    for mean, config in zip(means, params):
        config = str(config)
        if config in svm_parameters:
            svm_parameters[config].append(mean)
        else:
            svm_parameters[config] = [mean]


    # define dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
    # define models and parameters
    model = KNeighborsClassifier()
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    # define grid search
    grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_train, training_labels)
    # summarize results

    print(grid_result.best_params_)

 #   path = os.path.join(store_parameters, f"{i}.txt")

    means = grid_result.cv_results_['mean_test_score']
    print(max(means))
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']


    for mean, config in zip(means, params):
        config = str(config)
        if config in knn_paramters:
            knn_paramters[config].append(mean)
        else:
            knn_paramters[config] = [mean]


    X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
    # define models and parameters
    model = RandomForestClassifier()
    n_estimators = [10, 100, 1000]
    max_features = ['sqrt', 'log2']
    # define grid search
    grid = dict(n_estimators=n_estimators, max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_train, training_labels)
# summarize results
   # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print(grid_result.best_params_)


    means = grid_result.cv_results_['mean_test_score']
    print(max(means))
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, config in zip(means, params):
        config = str(config)
        if config in rf_paramters:
            rf_paramters[config].append(mean)
        else:
            rf_paramters[config] = [mean]


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
    grid_result = grid_search.fit(X_train, training_labels)
    # summarize results
    means = grid_result.cv_results_['mean_test_score']
    print(max(means))
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, config in zip(means, params):
        config = str(config)
        if config in xg_paramters:
            xg_paramters[config].append(mean)
        else:
            xg_paramters[config] = [mean]



    model = BaggingClassifier()
    max_samples = [0.05, 0.1, 0.2, 0.5]
    n_estimators = [10, 100, 1000]
    # define grid search
    grid = dict(n_estimators=n_estimators, max_samples=max_samples)
    cv = RepeatedStratifiedKFold(n_splits=9, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_train, training_labels)
    # summarize results
   # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print(grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    print(max(means))
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, config in zip(means, params):
        config = str(config)
        if config in bagg_paramters:
            bagg_paramters[config].append(mean)
        else:
            bagg_paramters[config] = [mean]



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





