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
np.random.seed(20)

feat_pth = sys.argv[1] # folder with saved feature files

def train_split(colombian, colombian_lab, czech, czech_lab, spain, spain_lab, german, german_lab):
    train_mat_data_point = np.concatenate([colombian, czech, spain, german], axis=0)

    train_data_label = np.concatenate([colombian_lab, czech_lab, spain_lab, german_lab],
                                      axis=0)

    return train_mat_data_point, train_data_label

def test_split(czech, czech_lab):
    train_mat_data_point = np.concatenate([czech], axis=0)
    train_data_label = np.concatenate([czech_lab], axis=0)
    return train_mat_data_point, train_data_label

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

def concat_train_data(colombian, colombian_lab, czech, czech_lab, spain, spain_lab, german, german_lab, english,
                      english_lab):
    train_mat_data_point = np.concatenate([colombian, czech, spain, german, english], axis=0)
    train_data_label = np.concatenate([colombian_lab, czech_lab, spain_lab, german_lab, english_lab], axis=0)
    return train_mat_data_point, train_data_label

def concat_test_data(czech, czech_lab):
    train_mat_data_point = np.concatenate([czech], axis=0)
    train_data_label = np.concatenate([czech_lab], axis=0)
    return train_mat_data_point, train_data_label


#% try two feats
for feat_used in ['xvector','trill']:
    print()
    print('----------------')
    print(feat_used)
    print('----------------')

    # get dataframe of all recordings with labels
    # ----------------
    nls = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/total_new_training.csv")
    nls = nls.drop(columns=['Unnamed: 0'])
    nls = nls.dropna()
    nls['filename'] = nls['names'] #%
    nls['task'] = [m.split("_", 3)[3].split(".wav")[0] for m in nls['names'].tolist()]
    task = ['CookieThief']
    nls = nls[nls['task'].isin(task)]

    nls = nls.sort_values(by=['names'])
    nls['id'] = [m.split("_ses")[0] for m in nls['names'].tolist()]
    nls.columns.tolist()

    label_seneca = pd.read_excel("/export/b15/afavaro/Book3.xlsx")
    label = label_seneca['Label'].tolist()
    speak = label_seneca['Participant I.D.'].tolist()
    spk2lab_ = {sp: lab for sp, lab in zip(speak, label)}
    speak2__ = nls['id'].tolist()

    etichettex = []
    for nome in speak2__:
        if nome in spk2lab_.keys():
            lav = spk2lab_[nome]
            etichettex.append(([nome, lav]))
        else:
            etichettex.append(([nome, 'Unknown']))

    label_new_ = []
    for e in etichettex:
        label_new_.append(e[1])
    nls['label'] = label_new_

    label = label_seneca['Age'].tolist()
    speak = label_seneca['Participant I.D.'].tolist()
    spk2lab_ = {sp: lab for sp, lab in zip(speak, label)}
    speak2__ = nls['id'].tolist()

    etichettex = []
    for nome in speak2__:
        if nome in spk2lab_.keys():
            lav = spk2lab_[nome]
            etichettex.append(([nome, lav]))
        else:
            etichettex.append(([nome, 'Unknown']))

    label_new_ = []
    for e in etichettex:
        label_new_.append(e[1])
    nls['age'] = label_new_

    TOT = nls.groupby('label')
    PD = TOT.get_group('PD')
    ctrl = TOT.get_group('CTRL')  # .grouby('ID')

    PD = PD[~PD.id.str.contains("NLS_116")]
    PD = PD[~PD.id.str.contains("NLS_34")]
    PD = PD[~PD.id.str.contains("NLS_35")]
    PD = PD[~PD.id.str.contains("NLS_33")]
    PD = PD[~PD.id.str.contains("NLS_12")]
    PD = PD[~PD.id.str.contains("NLS_21")]
    PD = PD[~PD.id.str.contains("NLS_20")]
    PD = PD[~PD.id.str.contains("NLS_12")]

    ctrl = ctrl[~ctrl.id.str.contains("PEC_4")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_5")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_9")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_14")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_15")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_16")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_17")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_18")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_19")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_23")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_25")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_29")]
    ctrl = ctrl[~ctrl.id.str.contains("PEC_35")]

    test = pd.concat([PD, ctrl])
    test = test.dropna()

    test = test.drop(columns=['age'])

    new = []
    for m in test['label'].tolist():
        if m == 'PD':
            new.append(1)
        if m == 'CTRL':
            new.append(0)
    test['label'] = new

    totale = test.drop(columns=['names', 'task'])
    nls = totale
    nls = nls.rename(columns={"label": "labels", 'names': 'id'})
    nls_cols = nls.columns.tolist()

    # ----------------
    colombian = pd.read_csv \
        ("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined_ling_tot.csv")

    colombian = colombian.dropna()
    colombian['filename'] = colombian['AudioFile'] #%
    colombian['names'] = [elem.split("_")[1] for elem in colombian.AudioFile.tolist()]
    colombian['labels'] = [elem.split("_")[0] for elem in colombian.AudioFile.tolist()]
    colombian['task'] = [elem.split("_")[2] for elem in colombian['AudioFile'].tolist()]
    task = ['monologue']
    colombian = colombian[colombian['task'].isin(task)]

    colombian = colombian.drop(columns=['Unnamed: 0'])

    new_lab = []
    for lab in colombian['labels'].tolist():
        if lab == "PD":
            new_lab.append(1)
        if lab == "CN":
            new_lab.append(0)
        if lab == "HC":
            new_lab.append(0)

    colombian['labels'] = new_lab
    colombian = colombian.rename(columns={'names': 'id'})
    # colombian['id'] = [elem.split("_")[1] for elem in colombian['AudioFile'].tolist()]
    colombian_cols = colombian.columns.tolist()

    # ----------------
    spain = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv")
    spain = spain.dropna()
    spain['filename'] = spain['AudioFile'] #%
    spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
    spain['task'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
    spain['id'] = [elem.split("_")[2].split("-")[0] for elem in spain['AudioFile'].tolist()]

    task = ['ESPONTANEA']
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
    german = pd.read_csv \
        ("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/final_data_frame_with_intensity.csv")
    german = german.drop(columns=['Unnamed: 0'])
    german['filename'] = german['AudioFile'] #%
    german['names'] = [elem.split("_")[1] for elem in german['AudioFile'].tolist()]
    german['labels'] = [elem.split("_")[0] for elem in german['AudioFile'].tolist()]
    german['task'] = [elem.split("_")[-2] for elem in german['AudioFile'].tolist()]

    task = ['monologue']
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
    # spain =pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/czech/final_data_experiments.csv")
    czech = pd.read_csv(
        "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/czech/final_data_experiments_updated.csv")
    czech['filename'] = czech['AudioFile'] #%
    czech['names'] = [elem.split("_")[1] for elem in czech.AudioFile.tolist()]
    czech['task'] = [elem.split("_")[2] for elem in czech['AudioFile'].tolist()]
    czech['labels'] = [elem.split("_")[0] for elem in czech.AudioFile.tolist()]
    task = ['monologue']
    czech = czech[czech['task'].isin(task)]
    czech = czech.drop(columns=['Unnamed: 0', 'AudioFile', 'task'])

    lab = []
    for m in czech['labels'].tolist():
        if m == "PD":
            lab.append(1)
        if m == 'CN':
            lab.append(0)
    czech['labels'] = lab
    czech = czech.dropna()
    
    czech = czech.rename(columns={"names": "id"})
    czech_clols = czech.columns.tolist()

    # ----------------

    # data organization
    one_inter = IntersecOfSets(german_cols, nls_cols, spain_cols)
    lista_to_keep = IntersecOfSets(one_inter, colombian_cols, czech_clols)

    nls = nls[nls.columns.intersection(lista_to_keep)]
    czech = czech[czech.columns.intersection(lista_to_keep)]
    colombian = colombian[colombian.columns.intersection(lista_to_keep)]
    german = german[german.columns.intersection(lista_to_keep)]
    spain = spain[spain.columns.intersection(lista_to_keep)]

    colombian = colombian.reindex(sorted(colombian.columns), axis=1)
    german = german.reindex(sorted(german.columns), axis=1)
    spain = spain.reindex(sorted(spain.columns), axis=1)
    nls = nls.reindex(sorted(nls.columns), axis=1)
    czech = czech.reindex(sorted(czech.columns), axis=1)
    # pdb.set_trace()

    # data normalization
    nls = preprocess_data_frame(nls)
    german = preprocess_data_frame(german)
    czech = preprocess_data_frame(czech)
    spain = preprocess_data_frame(spain)
    colombian = preprocess_data_frame(colombian)

    normalized_train_X_spain, y_train_spain, mean_spain, std_spain = normalize(spain)
    normalized_train_X_nls, y_train_nls, mean_nls, std_nls = normalize(nls)
    normalized_train_X_colombian, y_train_colombian, mean_colombian, std_colombian = normalize(colombian)
    normalized_train_X_czech, y_train_czech, mean_czech, std_czech = normalize(czech)
    normalized_train_X_german, y_train_german, mean_german, std_german = normalize(german)

    # 1- german test ---------------------------------
    means = np.mean(np.stack([mean_spain, mean_nls, mean_colombian, mean_czech], axis=1), axis=1)
    stds = np.mean(np.stack([std_spain, std_nls, std_colombian, std_czech], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(german, means, stds)

    training_data, training_labels = train_split(normalized_train_X_spain, y_train_spain, normalized_train_X_czech,
                                                y_train_czech,
                                                normalized_train_X_nls, y_train_nls, normalized_train_X_colombian,
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


    # 2- nls test ---------------------------------
    means = np.mean(np.stack([mean_spain, mean_german, mean_colombian, mean_czech], axis=1), axis=1)
    stds = np.mean(np.stack([std_spain, std_german, std_colombian, std_czech], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(nls, means, stds)

    training_data, training_labels = train_split(normalized_train_X_spain, y_train_spain, normalized_train_X_czech,
                                                y_train_czech,
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
    print('nls:')
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


    # 3- neurovoz test ---------------------------------
    means = np.mean(np.stack([mean_nls, mean_german, mean_colombian, mean_czech], axis=1), axis=1)
    stds = np.mean(np.stack([std_nls, std_german, std_colombian, std_czech], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(spain, means, stds)

    training_data, training_labels = train_split(normalized_train_X_nls, y_train_nls, normalized_train_X_czech,
                                                y_train_czech,
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


    # 4- czech test ---------------------------------

    means = np.mean(np.stack([mean_nls, mean_german, mean_colombian, mean_spain], axis=1), axis=1)
    stds = np.mean(np.stack([std_nls, std_german, std_colombian, std_spain], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(czech, means, stds)

    training_data, training_labels = train_split(normalized_train_X_nls, y_train_nls, normalized_train_X_spain,
                                                y_train_spain,
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
    print('czech:')
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


    # 5- GITA test ---------------------------------

    means = np.mean(np.stack([mean_nls, mean_german, mean_czech, mean_spain], axis=1), axis=1)
    stds = np.mean(np.stack([std_nls, std_german, std_czech, std_spain], axis=1), axis=1)

    normalized_test_X, normalized_y_test = normalize_test(colombian, means, stds)

    training_data, training_labels = train_split(normalized_train_X_nls, y_train_nls, normalized_train_X_spain,
                                                y_train_spain,
                                                normalized_train_X_german, y_train_german, normalized_train_X_czech,
                                                y_train_czech)

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