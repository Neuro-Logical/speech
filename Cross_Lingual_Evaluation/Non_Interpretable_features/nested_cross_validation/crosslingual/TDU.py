# monologue: nls-cookie, german/GITA/czech-monologue, spain-ESPONTANEA
# readpassage: nls-RP(poem+rainbow), german/GITA/czech-readtext, ita-RP(B1+B2)
# TDU: GITA-TDU, german-concatenateread, spain-concatenateread, ita-FB
import pdb
from PCA_PLDA_EER_Classifier import PCA_PLDA_EER_Classifier
from statistics import mode
import random
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import LeaveOneOut
random.seed(20)

feat_pth = sys.argv[1] # folder with saved feature files

def test_split(czech, czech_lab):
    train_mat_data_point = np.concatenate([czech], axis=0)
    train_data_label = np.concatenate([czech_lab], axis=0)
    return train_mat_data_point, train_data_label

def train_split(colombian, colombian_lab, czech, czech_lab, spain, spain_lab):
    train_mat_data_point = np.concatenate([colombian, czech, spain], axis=0)

    train_data_label = np.concatenate([colombian_lab, czech_lab, spain_lab],
                                      axis=0)

    return train_mat_data_point, train_data_label

def concat_train_data(colombian, colombian_lab, czech, czech_lab, spain, spain_lab):
    train_mat_data_point = np.concatenate([colombian, czech, spain], axis=0)
    train_data_label = np.concatenate([colombian_lab, czech_lab, spain_lab], axis=0)
    return train_mat_data_point, train_data_label

def concat_test_data(czech, czech_lab):
    train_mat_data_point = np.concatenate([czech], axis=0)
    train_data_label = np.concatenate([czech_lab], axis=0)
    return train_mat_data_point, train_data_label

#%
def preprocess_data_frame(data_frame):
    data_fold = np.array(()) #%
    #% extract features from files
    for index, row in data_frame.iterrows():
        name_no_exten = os.path.splitext(row['filename'])[0] # remove file extension if any
        label_row = row['labels']
        feat = np.load(feat_pth+feat_used+'/' + name_no_exten + '.npy')
        feat = np.append(feat,label_row) # attach label to the end of array [1, feat dim + 1]
        data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat

    # # nomi = data_frame['id'].tolist()
    # lab = data_frame['labels'].tolist()
    # data_frame = data_frame.drop(columns=['labels', 'id'])
    # # data_frame['id'] = nomi
    # data_frame['labels'] = lab

    return data_fold #data_frame

def normalize(train_set):
    # train_set = train_set.to_numpy() #%
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1:]

    control_group = train_set[train_set[:, -1] == 0]

    control_group = control_group[:, :-1]  # remove labels from features CNs
    median = np.median(control_group, axis=0)
    std = np.std(control_group, axis=0)
    #
    y_train = y_train.ravel()
    x_train = x_train.astype('float')
    #
    normalized_x_train = (x_train - median) / (std + 0.01)

    return normalized_x_train, y_train, median, std

def normalize_test(test, median, std):
    # train_set = test.to_numpy() #%
    train_set = test
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1:]
    y_train = y_train.ravel()
    x_train = x_train.astype('float')
    #
    normalized_x_train = (x_train - median) / (std + 0.01)

    return normalized_x_train, y_train

def IntersecOfSets(arr1, arr2, arr3):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)

    set1 = s1.intersection(s2)  # [80, 20, 100]

    result_set = set1.intersection(s3)

    # Converts resulting set to list
    final_list = list(result_set)
    return final_list

def IntersecOftwo(arr1, arr2):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    set1 = s1.intersection(s2)
    final_list = list(set1)
    return final_list


#% try two feats
for feat_used in ['xvector','trill']:
    print()
    print('----------------')
    print(feat_used)
    print('----------------')

    # get dataframe of all recordings with labels
    # ----------------
    colombian = pd.read_csv(
        "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined_ling_tot.csv")

    # spain = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined.csv")
    colombian = colombian.dropna()
    colombian['filename'] = colombian['AudioFile'] #%
    colombian['task'] = [elem.split("_")[2] for elem in colombian['AudioFile'].tolist()]
    task = ['TDU']
    colombian = colombian[colombian['task'].isin(task)]
    colombian = colombian.drop(columns=['Unnamed: 0'])
    colombian['labels'] = [elem.split("_")[0] for elem in colombian['AudioFile'].tolist()]

    colombian['id'] = [elem.split("_")[1] for elem in colombian['AudioFile'].tolist()]
    colombian = colombian.drop(columns=['AudioFile', 'task'])

    new_lab = []
    for lab in colombian['labels'].tolist():
        if lab == "PD":
            new_lab.append(1)
        if lab == "CN":
            new_lab.append(0)
        if lab == "HC":
            new_lab.append(0)

    colombian['labels'] = new_lab

    colombian_cols = colombian.columns.tolist()

    # ----------------
    spain = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv")
    spain = spain.dropna()
    spain['filename'] = spain['AudioFile'] #%
    spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
    spain['task'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
    spain['id'] = [elem.split("_")[2].split("-")[0] for elem in spain['AudioFile'].tolist()]

    task = ['concatenateread']
    spain = spain[spain['task'].isin(task)]
    spain = spain.drop(columns=['Unnamed: 0', 'AudioFile', 'task'])

    lab = []
    for m in spain['labels'].tolist():
        if m == 'PD':
            lab.append(1)
        if m == 'HC':
            lab.append(0)
    spain['labels'] = lab

    spain_cols = spain.columns.tolist()

    # ----------------
    german = pd.read_csv(
        "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/final_data_frame_with_intensity.csv")
    german = german.drop(columns=['Unnamed: 0'])
    german['filename'] = german['AudioFile'] #%
    german['names'] = [elem.split("_")[1] for elem in german['AudioFile'].tolist()]
    german['labels'] = [elem.split("_")[0] for elem in german['AudioFile'].tolist()]
    german['task'] = [elem.split("_")[-2] for elem in german['AudioFile'].tolist()]

    task = ['concatenateread']
    german = german[german['task'].isin(task)]
    german = german.drop(columns=['AudioFile', 'task'])

    lab = []
    for m in german['labels'].tolist():
        if m == "PD":
            lab.append(1)
        if m == 'CN':
            lab.append(0)
    german['labels'] = lab
    german = german.dropna()

    german = german.rename(columns={"names": "id"})
    german_cols = german.columns.tolist()

    # ----------------
    italian = pd.read_csv(
        "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/tot_experiments_ling_fin.csv")

    # italian = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/tot_experiments.csv")
    italian['labels'] = [m.split("_")[0] for m in italian.AudioFile.tolist()]
    italian['task'] = [elem.split("_")[3][:2] for elem in italian['AudioFile'].tolist()]
    task = ['FB']

    italian = italian[italian['task'].isin(task)]

    lab = []
    for m in italian['labels']:
        if m == 'PD':
            lab.append(1)
        if m == 'CN':
            lab.append(0)

    italian['labels'] = lab

    names = [m.split("_", -1)[1] for m in italian.AudioFile.tolist()]
    surname = [m.split("_", -1)[2] for m in italian.AudioFile.tolist()]
    totale_names = []
    for i in zip(names, surname):
        totale_names.append(i[0] + i[1])
    italian['id'] = totale_names
    italian['filename'] = italian['AudioFile'] #%
    italian = italian.drop(columns=['Unnamed: 0', 'AudioFile', 'task'])
    italian_cols = italian.columns.tolist()

    # ----------------

    # data organization
    one_inter = IntersecOftwo(german_cols, italian_cols)
    lista_to_keep = IntersecOfSets(one_inter, colombian_cols, spain_cols)

    colombian = colombian[colombian.columns.intersection(lista_to_keep)]
    german = german[german.columns.intersection(lista_to_keep)]
    italian = italian[italian.columns.intersection(lista_to_keep)]
    spain = spain[spain.columns.intersection(lista_to_keep)]

    colombian = colombian.reindex(sorted(colombian.columns), axis=1)
    german = german.reindex(sorted(german.columns), axis=1)
    spain = spain.reindex(sorted(spain.columns), axis=1)
    italian = italian.reindex(sorted(italian.columns), axis=1)


    # data normalization
    german = preprocess_data_frame(german)
    italian = preprocess_data_frame(italian)
    colombian = preprocess_data_frame(colombian)
    spain = preprocess_data_frame(spain)

    normalized_train_X_german, y_train_german, mean_german, std_german = normalize(german)
    normalized_train_X_italian, y_train_italian, mean_italian, std_italian = normalize(italian)
    normalized_train_X_spain, y_train_spain, mean_spain, std_spain = normalize(spain)
    normalized_train_X_colombian, y_train_colombian, mean_colombian, std_colombian = normalize(colombian)

    # 1- german test ---------------------------------
    means = np.mean(np.stack([mean_italian, mean_spain, mean_colombian], axis=1), axis=1)
    stds = np.mean(np.stack([std_italian, std_spain, std_colombian], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(german, means, stds)

    training_data, training_labels = train_split(normalized_train_X_italian, y_train_italian, 
                                                normalized_train_X_spain, y_train_spain, normalized_train_X_colombian,
                                                y_train_colombian)

    test_data, test_labels = test_split(normalized_test_X, normalized_y_test)

    # cross-validation - hyperparameter search
    tuned_params = {"PCA_n" : [5,10,15,20,25,30,35,40,45,50,55]} #%
    model = PCA_PLDA_EER_Classifier(normalize=0)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    grid_search = GridSearchCV(estimator=model, param_grid=tuned_params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)

    grid_result = grid_search.fit(training_data, training_labels)
    print(grid_result.best_params_)
    best_param = grid_result.best_params_['PCA_n']

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print(means)

    # test
    y_test = test_labels.tolist()

    model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
    model.fit(training_data, training_labels)
    grid_predictions = model.predict(test_data)
    grid_test_scores = model.predict_scores_list(test_data)

    test_scores = grid_test_scores[:,0].tolist()

    # report
    print('german:')
    print(classification_report(y_test, grid_predictions, output_dict=False))
    print(confusion_matrix(y_test, grid_predictions))

    tn, fp, fn, tp = confusion_matrix(test_labels, grid_predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(test_labels,test_scores))
    print('----------')


    # 2- neurovoz test ---------------------------------
    means = np.mean(np.stack([mean_italian, mean_german, mean_colombian], axis=1), axis=1)
    stds = np.mean(np.stack([std_italian, std_german, std_colombian], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(spain, means, stds)

    training_data, training_labels = train_split(normalized_train_X_italian, y_train_italian, 
                                                normalized_train_X_german, y_train_german, normalized_train_X_colombian,
                                                y_train_colombian)

    test_data, test_labels = test_split(normalized_test_X, normalized_y_test)

    # cross-validation - hyperparameter search
    tuned_params = {"PCA_n" : [5,10,15,20,25,30,35,40,45,50,55]} #%
    model = PCA_PLDA_EER_Classifier(normalize=0)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    grid_search = GridSearchCV(estimator=model, param_grid=tuned_params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)

    grid_result = grid_search.fit(training_data, training_labels)
    print(grid_result.best_params_)
    best_param = grid_result.best_params_['PCA_n']

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print(means)

    # test
    y_test = test_labels.tolist()

    model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
    model.fit(training_data, training_labels)
    grid_predictions = model.predict(test_data)
    grid_test_scores = model.predict_scores_list(test_data)

    test_scores = grid_test_scores[:,0].tolist()

    # report
    print('Neurovoz:')
    print(classification_report(y_test, grid_predictions, output_dict=False))
    print(confusion_matrix(test_labels, grid_predictions))

    tn, fp, fn, tp = confusion_matrix(test_labels, grid_predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(test_labels,test_scores))
    print('----------')


    # 3- italian test ---------------------------------
    means = np.mean(np.stack([mean_spain, mean_german, mean_colombian], axis=1), axis=1)
    stds = np.mean(np.stack([std_spain, std_german, std_colombian], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(italian, means, stds)

    training_data, training_labels = train_split(normalized_train_X_spain, y_train_spain, 
                                                normalized_train_X_german, y_train_german, normalized_train_X_colombian,
                                                y_train_colombian)

    test_data, test_labels = test_split(normalized_test_X, normalized_y_test)

    # cross-validation - hyperparameter search
    tuned_params = {"PCA_n" : [5,10,15,20,25,30,35,40,45,50,55]} #%
    model = PCA_PLDA_EER_Classifier(normalize=0)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    grid_search = GridSearchCV(estimator=model, param_grid=tuned_params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)

    grid_result = grid_search.fit(training_data, training_labels)
    print(grid_result.best_params_)
    best_param = grid_result.best_params_['PCA_n']

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print(means)

    # test
    y_test = test_labels.tolist()

    model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
    model.fit(training_data, training_labels)
    grid_predictions = model.predict(test_data)
    grid_test_scores = model.predict_scores_list(test_data)

    test_scores = grid_test_scores[:,0].tolist()

    # report
    print('italian:')
    print(classification_report(y_test, grid_predictions, output_dict=False))
    print(confusion_matrix(test_labels, grid_predictions))

    tn, fp, fn, tp = confusion_matrix(test_labels, grid_predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(test_labels,test_scores))
    print('----------')


    # 4- GITA test ---------------------------------
    means = np.mean(np.stack([mean_spain, mean_german, mean_italian], axis=1), axis=1)
    stds = np.mean(np.stack([std_spain, std_german, std_italian], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(colombian, means, stds)

    training_data, training_labels = train_split(normalized_train_X_spain, y_train_spain, normalized_train_X_italian,
                                                y_train_italian,
                                                normalized_train_X_german, y_train_german)

    test_data, test_labels = test_split(normalized_test_X, normalized_y_test)

    # cross-validation - hyperparameter search
    tuned_params = {"PCA_n" : [5,10,15,20,25,30,35,40,45,50,55]} #%
    model = PCA_PLDA_EER_Classifier(normalize=0)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    grid_search = GridSearchCV(estimator=model, param_grid=tuned_params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)

    grid_result = grid_search.fit(training_data, training_labels)
    print(grid_result.best_params_)
    best_param = grid_result.best_params_['PCA_n']

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print(means)

    # test
    y_test = test_labels.tolist()

    model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
    model.fit(training_data, training_labels)
    grid_predictions = model.predict(test_data)
    grid_test_scores = model.predict_scores_list(test_data)

    test_scores = grid_test_scores[:,0].tolist()

    # report
    print('GITA:')
    print(classification_report(y_test, grid_predictions, output_dict=False))
    print(confusion_matrix(test_labels, grid_predictions))

    tn, fp, fn, tp = confusion_matrix(test_labels, grid_predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(test_labels,test_scores))
    print('----------')