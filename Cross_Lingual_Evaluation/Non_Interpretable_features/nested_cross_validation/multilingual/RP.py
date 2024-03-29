#!/usr/bin/env python
# coding: utf-8
import pdb
from PCA_PLDA_EER_Classifier import PCA_PLDA_EER_Classifier
from statistics import mode
import random
random.seed(20)

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import LeaveOneOut

# multi
# monologue: nls-cookie, german/GITA/czech-monologue, spain-ESPONTANEA
# readpassage: nls-RP(poem+rainbow), german/GITA/czech-readtext, ita-RP(B1+B2)
# TDU: GITA-TDU, german-concatenateread, spain-concatenateread, ita-FB
#
# !!!!!!!!!!!!!!!!!!!!!!!!
# set test_only = 0 if doing hyperparameter search. 
# uncomment best_param part under "for feat_used in ['xvector','trill']:", 
# set hyperparamters there, and set test_only = 1 if skipping hyperparameter search 
# !!!!!!!!!!!!!!!!!!!!!!!!
test_only = 0
feat_pth = sys.argv[1] # folder with saved feature files

def normalize(train_split, test_split):
    train_set = train_split
    test_set = test_split

    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]

    control_group = train_set[train_set[:, -1] == 0]
    control_group = control_group[:, :-1]  # remove labels from features CNs

    median = np.median(control_group, axis=0)
    std = np.std(control_group, axis=0)

    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')

    normalized_train_X = (X_train - median) / (std + 0.01)
    normalized_test_X = (X_test - median) / (std + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test

def preprocess_data_frame(data_frame):
    nomi = data_frame['id'].tolist()
    lab = data_frame['labels'].tolist()
    data_frame = data_frame.drop(columns=['id', 'labels'])
    data_frame['id'] = nomi
    data_frame['labels'] = lab

    return data_frame

def get_n_folds(arrayOfSpeaker):
    # generate speakers for outer folds
    data = list(arrayOfSpeaker)  # list(range(len(arrayOfSpeaker)))
    num_of_folds = 10
    n_folds = []
    for i in range(num_of_folds):
        n_folds.append(data[int(i * len(data) / num_of_folds):int((i + 1) * len(data) / num_of_folds)])
    return n_folds

def create_n_folds(data_frame):
    # generate outer folds data
    data = []
    folds = []

    data_grouped = data_frame.groupby('labels')
    ctrl_ = data_grouped.get_group(0)
    pd_ = data_grouped.get_group(1)

    arrayOfSpeaker_cn = ctrl_['id'].unique()
    random.shuffle(arrayOfSpeaker_cn)

    arrayOfSpeaker_pd = pd_['id'].unique()
    random.shuffle(arrayOfSpeaker_pd)

    cn_sps = get_n_folds(arrayOfSpeaker_cn)
    pd_sps = get_n_folds(arrayOfSpeaker_pd)

    for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
        data.append(cn_sp + pd_sp)
    n_folds = sorted(data, key=len, reverse=True)

    #%
    folds = []
    for i in n_folds:
        data_fold = np.array(()) #%
        data_i = data_frame[data_frame["id"].isin(i)]
        #% extract features from files
        for index, row in data_i.iterrows():
            name_no_exten = os.path.splitext(row['filename'])[0] # remove file extension if any
            label_row = row['labels']
            feat = np.load(feat_pth+feat_used+'/' + name_no_exten + '.npy')
            feat = np.append(feat,label_row) # attach label to the end of array [1, feat dim + 1]
            data_fold = np.vstack((data_fold, feat)) if data_fold.size else feat
        folds.append(data_fold)

    # for i in n_folds:
    #     data_i = data_frame[data_frame["id"].isin(i)]
    #     data_i = data_i.drop(columns=['id'])
    #     folds.append((data_i).to_numpy())

    return folds

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

def train_split(colombian, colombian_lab, czech, czech_lab, spain, spain_lab, german, german_lab,
                english, english_lab):
    train_mat_data_point = np.concatenate([colombian, czech, spain, german, english], axis=0)

    train_data_label = np.concatenate([colombian_lab, czech_lab, spain_lab, german_lab, english_lab],
                                      axis=0)

    return train_mat_data_point, train_data_label

def test_split(czech, czech_lab):
    train_mat_data_point = np.concatenate([czech], axis=0)

    train_data_label = np.concatenate([czech_lab], axis=0)

    return train_mat_data_point, train_data_label


#% try two feats
best_param_init =  {
            'xvector': 55,
            'trill': 50,
            }
# for feat_used in ['xvector','trill']:
for feat_used in ['wav2vec2/hidden8','wav2vec2/hidden6','wav2vec2/hidden5','wav2vec2/hidden4','wav2vec2/hidden3',
'wav2vec2/hidden2','wav2vec2/hidden1','wav2vec2/hidden0','hubert/hidden8','hubert/hidden6','hubert/hidden5','hubert/hidden4','hubert/hidden3',
'hubert/hidden2','hubert/hidden1','hubert/hidden0','xvector','trill']:
    if feat_used in best_param_init.keys():
        best_param = best_param_init[feat_used]
    print()
    print('----------------')
    print(feat_used)
    print('----------------')

    # get dataframe of all recordings with labels
    # ----------------
    nls = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/Data_frame_RP.csv")
    nls = nls.drop(columns=['Unnamed: 0'])
    nls = nls.dropna()
    nls['filename'] = nls['AudioFile'] #%
    nls = nls.sort_values(by=['AudioFile'])
    nls['id'] = [m.split("_RP")[0] for m in nls['AudioFile'].tolist()]

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
    #ctrl = ctrl[ctrl['age'] > 68]

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
    totale = test.drop(columns=['AudioFile'])
    nls = totale.rename(columns={"label": "labels", 'names':'id'})
    nls_cols = nls.columns.tolist()

    # ----------------
    colombian = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined_ling_tot.csv")

    #spain = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined.csv")
    colombian = colombian.dropna()
    colombian['filename'] = colombian['AudioFile'] #%
    colombian['task'] = [elem.split("_")[2] for elem in colombian['AudioFile'].tolist()]
    task = ['readtext']
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
    #colombian = colombian.rename(columns={'names':'id'})
    #colombian['id'] = [elem.split("_")[1] for elem in colombian['AudioFile'].tolist()]
    colombian_cols = colombian.columns.tolist()

    # ----------------
    german = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/final_data_frame_with_intensity.csv")
    german = german.drop(columns=['Unnamed: 0'])
    german['filename'] = german['AudioFile'] #%
    german['names'] = [elem.split("_")[1] for elem in german['AudioFile'].tolist()]
    german['labels'] = [elem.split("_")[0] for elem in german['AudioFile'].tolist()]
    german['task'] = [elem.split("_")[-2] for elem in german['AudioFile'].tolist()]
    task = ['readtext']
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
    #german

    # ----------------
    czech =pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/Czech/final_data_experiments_updated.csv")
    czech['filename'] = czech['AudioFile'] #%
    czech['names'] =  [elem.split("_")[1] for elem in czech.AudioFile.tolist()]
    czech['task'] = [elem.split("_")[2] for elem in czech['AudioFile'].tolist()]
    czech['labels'] = [elem.split("_")[0] for elem in czech.AudioFile.tolist()]
    task = ['readtext']
    czech = czech[czech['task'].isin(task)]
    czech = czech.drop(columns=['Unnamed: 0',  'AudioFile', 'task'])
    lab =[]
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
    italian = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/RP_data_frame.csv")

    #italian = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/tot_experiments.csv")
    italian['labels'] = [m.split("_")[0] for m in italian.AudioFile.tolist()]
    italian['filename'] = italian['AudioFile'] #%

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

    italian = italian.drop(columns=['Unnamed: 0', 'AudioFile'])
    italian_cols = italian.columns.tolist()
    #italian

    # ----------------

    # data organization
    one_inter = IntersecOfSets(german_cols, nls_cols, italian_cols)
    lista_to_keep = IntersecOfSets(one_inter, colombian_cols, czech_clols)

    nls = nls[nls.columns.intersection(lista_to_keep)]
    czech = czech[czech.columns.intersection(lista_to_keep)]
    colombian = colombian[colombian.columns.intersection(lista_to_keep)]
    german = german[german.columns.intersection(lista_to_keep)]
    italian = italian[italian.columns.intersection(lista_to_keep)]

    colombian = colombian.reindex(sorted(colombian.columns), axis=1)
    german = german.reindex(sorted(german.columns), axis=1)
    nls = nls.reindex(sorted(nls.columns), axis=1)
    czech = czech.reindex(sorted(czech.columns), axis=1)
    italian = italian.reindex(sorted(italian.columns), axis=1)


    # generate train/test splits for outer folds
    nls = preprocess_data_frame(nls)
    nls_folds = create_n_folds(nls)

    data_train_1_nls = np.concatenate(nls_folds[:9])
    data_test_1_nls = np.concatenate(nls_folds[-1:])

    data_train_2_nls = np.concatenate(nls_folds[1:])
    data_test_2_nls = np.concatenate(nls_folds[:1])

    data_train_3_nls = np.concatenate(nls_folds[2:] + nls_folds[:1])
    data_test_3_nls = np.concatenate(nls_folds[1:2])

    data_train_4_nls = np.concatenate(nls_folds[3:] + nls_folds[:2])
    data_test_4_nls = np.concatenate(nls_folds[2:3])

    data_train_5_nls = np.concatenate(nls_folds[4:] + nls_folds[:3])
    data_test_5_nls = np.concatenate(nls_folds[3:4])

    data_train_6_nls = np.concatenate(nls_folds[5:] + nls_folds[:4])
    data_test_6_nls = np.concatenate(nls_folds[4:5])

    data_train_7_nls = np.concatenate(nls_folds[6:] + nls_folds[:5])
    data_test_7_nls = np.concatenate(nls_folds[5:6])

    data_train_8_nls = np.concatenate(nls_folds[7:] + nls_folds[:6])
    data_test_8_nls = np.concatenate(nls_folds[6:7])

    data_train_9_nls = np.concatenate(nls_folds[8:] + nls_folds[:7])
    data_test_9_nls = np.concatenate(nls_folds[7:8])

    data_train_10_nls = np.concatenate(nls_folds[9:] + nls_folds[:8])
    data_test_10_nls = np.concatenate(nls_folds[8:9])

    # ----------------
    german = preprocess_data_frame(german)
    german_folds = create_n_folds(german)

    data_train_1_german = np.concatenate(german_folds[:9])
    data_test_1_german = np.concatenate(german_folds[-1:])

    data_train_2_german = np.concatenate(german_folds[1:])
    data_test_2_german = np.concatenate(german_folds[:1])

    data_train_3_german = np.concatenate(german_folds[2:] + german_folds[:1])
    data_test_3_german = np.concatenate(german_folds[1:2])

    data_train_4_german = np.concatenate(german_folds[3:] + german_folds[:2])
    data_test_4_german = np.concatenate(german_folds[2:3])

    data_train_5_german = np.concatenate(german_folds[4:] + german_folds[:3])
    data_test_5_german = np.concatenate(german_folds[3:4])

    data_train_6_german = np.concatenate(german_folds[5:] + german_folds[:4])
    data_test_6_german = np.concatenate(german_folds[4:5])

    data_train_7_german = np.concatenate(german_folds[6:] + german_folds[:5])
    data_test_7_german = np.concatenate(german_folds[5:6])

    data_train_8_german = np.concatenate(german_folds[7:] + german_folds[:6])
    data_test_8_german = np.concatenate(german_folds[6:7])

    data_train_9_german = np.concatenate(german_folds[8:] + german_folds[:7])
    data_test_9_german = np.concatenate(german_folds[7:8])

    data_train_10_german = np.concatenate(german_folds[9:] + german_folds[:8])
    data_test_10_german = np.concatenate(german_folds[8:9])

    # ----------------
    czech = preprocess_data_frame(czech)
    czech_folds = create_n_folds(czech)

    data_train_1_czech = np.concatenate(czech_folds[:9])
    data_test_1_czech = np.concatenate(czech_folds[-1:])

    data_train_2_czech = np.concatenate(czech_folds[1:])
    data_test_2_czech = np.concatenate(czech_folds[:1])

    data_train_3_czech = np.concatenate(czech_folds[2:] + czech_folds[:1])
    data_test_3_czech = np.concatenate(czech_folds[1:2])

    data_train_4_czech = np.concatenate(czech_folds[3:] + czech_folds[:2])
    data_test_4_czech = np.concatenate(czech_folds[2:3])

    data_train_5_czech = np.concatenate(czech_folds[4:] + czech_folds[:3])
    data_test_5_czech = np.concatenate(czech_folds[3:4])

    data_train_6_czech = np.concatenate(czech_folds[5:] + czech_folds[:4])
    data_test_6_czech = np.concatenate(czech_folds[4:5])

    data_train_7_czech = np.concatenate(czech_folds[6:] + czech_folds[:5])
    data_test_7_czech = np.concatenate(czech_folds[5:6])

    data_train_8_czech = np.concatenate(czech_folds[7:] + czech_folds[:6])
    data_test_8_czech = np.concatenate(czech_folds[6:7])

    data_train_9_czech = np.concatenate(czech_folds[8:] + czech_folds[:7])
    data_test_9_czech = np.concatenate(czech_folds[7:8])

    data_train_10_czech = np.concatenate(czech_folds[9:] + czech_folds[:8])
    data_test_10_czech = np.concatenate(czech_folds[8:9])

    # ----------------
    colombian = preprocess_data_frame(colombian)
    colombian_folds = create_n_folds(colombian)

    data_train_1_colombian = np.concatenate(colombian_folds[:9])
    data_test_1_colombian = np.concatenate(colombian_folds[-1:])

    data_train_2_colombian = np.concatenate(colombian_folds[1:])
    data_test_2_colombian = np.concatenate(colombian_folds[:1])

    data_train_3_colombian = np.concatenate(colombian_folds[2:] + colombian_folds[:1])
    data_test_3_colombian = np.concatenate(colombian_folds[1:2])

    data_train_4_colombian = np.concatenate(colombian_folds[3:] + colombian_folds[:2])
    data_test_4_colombian = np.concatenate(colombian_folds[2:3])

    data_train_5_colombian = np.concatenate(colombian_folds[4:] + colombian_folds[:3])
    data_test_5_colombian = np.concatenate(colombian_folds[3:4])

    data_train_6_colombian = np.concatenate(colombian_folds[5:] + colombian_folds[:4])
    data_test_6_colombian = np.concatenate(colombian_folds[4:5])

    data_train_7_colombian = np.concatenate(colombian_folds[6:] + colombian_folds[:5])
    data_test_7_colombian = np.concatenate(colombian_folds[5:6])

    data_train_8_colombian = np.concatenate(colombian_folds[7:] + colombian_folds[:6])
    data_test_8_colombian = np.concatenate(colombian_folds[6:7])

    data_train_9_colombian = np.concatenate(colombian_folds[8:] + colombian_folds[:7])
    data_test_9_colombian = np.concatenate(colombian_folds[7:8])

    data_train_10_colombian = np.concatenate(colombian_folds[9:] + colombian_folds[:8])
    data_test_10_colombian = np.concatenate(colombian_folds[8:9])

    # ----------------
    italian = preprocess_data_frame(italian)
    italian_folds = create_n_folds(italian)

    data_train_1_italian = np.concatenate(italian_folds[:9])
    data_test_1_italian = np.concatenate(italian_folds[-1:])

    data_train_2_italian = np.concatenate(italian_folds[1:])
    data_test_2_italian = np.concatenate(italian_folds[:1])

    data_train_3_italian = np.concatenate(italian_folds[2:] + italian_folds[:1])
    data_test_3_italian = np.concatenate(italian_folds[1:2])

    data_train_4_italian = np.concatenate(italian_folds[3:] + italian_folds[:2])
    data_test_4_italian = np.concatenate(italian_folds[2:3])

    data_train_5_italian = np.concatenate(italian_folds[4:] + italian_folds[:3])
    data_test_5_italian = np.concatenate(italian_folds[3:4])

    data_train_6_italian = np.concatenate(italian_folds[5:] + italian_folds[:4])
    data_test_6_italian = np.concatenate(italian_folds[4:5])

    data_train_7_italian = np.concatenate(italian_folds[6:] + italian_folds[:5])
    data_test_7_italian = np.concatenate(italian_folds[5:6])

    data_train_8_italian = np.concatenate(italian_folds[7:] + italian_folds[:6])
    data_test_8_italian = np.concatenate(italian_folds[6:7])

    data_train_9_italian = np.concatenate(italian_folds[8:] + italian_folds[:7])
    data_test_9_italian = np.concatenate(italian_folds[7:8])

    data_train_10_italian = np.concatenate(italian_folds[9:] + italian_folds[:8])
    data_test_10_italian = np.concatenate(italian_folds[8:9])


    # inner folds cross-validation - hyperparameter search
    if test_only == 0:
        best_params = []
        for i in range(1, 11):

            print(i)

            normalized_train_X_nls, normalized_test_X_nls, y_train_nls, y_test_nls = normalize(eval(f"data_train_{i}_nls"), eval(f"data_test_{i}_nls"))
            normalized_train_X_italian, normalized_test_X_italian, y_train_italian, y_test_italian = normalize(eval(f"data_train_{i}_italian"), eval(f"data_test_{i}_italian"))
            normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
            normalized_train_X_czech, normalized_test_X_czech, y_train_czech, y_test_czech = normalize(eval(f"data_train_{i}_czech"), eval(f"data_test_{i}_czech"))
            normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

            training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_czech, y_train_czech, normalized_train_X_german,
                                                        y_train_german, normalized_train_X_italian, y_train_italian, normalized_train_X_nls, y_train_nls)

            tuned_params = {"PCA_n" : [5,10,15,20,25,30,35,40,45,50,55]}
            model = PCA_PLDA_EER_Classifier(normalize=0)

            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
            # cv = LeaveOneOut()

            grid_search = GridSearchCV(estimator=model, param_grid=tuned_params, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
            
            grid_result = grid_search.fit(training_data, training_labels)
            print(grid_result.best_params_)

            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            print(means)

            best_params.append(grid_result.best_params_['PCA_n'])

        # get best params
        print('**********best pca n:')
        best_param = mode(best_params)
        print(best_param)
        print()

    # Test - czech -----------------------
    predictions = []
    truth = []
    test_scores = []
    for i in range(1, 11):

        normalized_train_X_nls, normalized_test_X_nls, y_train_nls, y_test_nls = normalize(eval(f"data_train_{i}_nls"), eval(f"data_test_{i}_nls"))
        normalized_train_X_italian, normalized_test_X_italian, y_train_italian, y_test_italian = normalize(eval(f"data_train_{i}_italian"), eval(f"data_test_{i}_italian"))
        normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
        normalized_train_X_czech, normalized_test_X_czech, y_train_czech, y_test_czech = normalize(eval(f"data_train_{i}_czech"), eval(f"data_test_{i}_czech"))
        normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

        training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_czech, y_train_czech, normalized_train_X_german,
                                                    y_train_german, normalized_train_X_italian, y_train_italian, normalized_train_X_nls, y_train_nls)

        y_test_czech = y_test_czech.tolist()

        model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
        model.fit(training_data, training_labels)
        grid_predictions = model.predict(normalized_test_X_czech)
        grid_test_scores = model.predict_scores_list(normalized_test_X_czech)

        predictions = predictions + grid_predictions
        truth = truth + y_test_czech
        test_scores += grid_test_scores[:,0].tolist()

    # report
    print('Czech:')
    print(classification_report(truth, predictions, output_dict=False))
    print(confusion_matrix(truth, predictions))

    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(truth,test_scores))
    print('----------')

    # Test - nls -----------------------
    predictions = []
    truth = []
    test_scores = []
    for i in range(1, 11):

        normalized_train_X_nls, normalized_test_X_nls, y_train_nls, y_test_nls = normalize(eval(f"data_train_{i}_nls"), eval(f"data_test_{i}_nls"))
        normalized_train_X_italian, normalized_test_X_italian, y_train_italian, y_test_italian = normalize(eval(f"data_train_{i}_italian"), eval(f"data_test_{i}_italian"))
        normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
        normalized_train_X_czech, normalized_test_X_czech, y_train_czech, y_test_czech = normalize(eval(f"data_train_{i}_czech"), eval(f"data_test_{i}_czech"))
        normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

        training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_czech, y_train_czech, normalized_train_X_german,
                                                    y_train_german, normalized_train_X_italian, y_train_italian, normalized_train_X_nls, y_train_nls)

        y_test_nls = y_test_nls.tolist()

        model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
        model.fit(training_data, training_labels)
        grid_predictions = model.predict(normalized_test_X_nls)
        grid_test_scores = model.predict_scores_list(normalized_test_X_nls)

        predictions = predictions + grid_predictions
        truth = truth + y_test_nls
        test_scores += grid_test_scores[:,0].tolist()

    # report
    print('NLS:')
    print(classification_report(truth, predictions, output_dict=False))
    print(confusion_matrix(truth, predictions))

    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(truth,test_scores))
    print('----------')

    # Test - italian -----------------------
    predictions = []
    truth = []
    test_scores = []
    for i in range(1, 11):

        normalized_train_X_nls, normalized_test_X_nls, y_train_nls, y_test_nls = normalize(eval(f"data_train_{i}_nls"), eval(f"data_test_{i}_nls"))
        normalized_train_X_italian, normalized_test_X_italian, y_train_italian, y_test_italian = normalize(eval(f"data_train_{i}_italian"), eval(f"data_test_{i}_italian"))
        normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
        normalized_train_X_czech, normalized_test_X_czech, y_train_czech, y_test_czech = normalize(eval(f"data_train_{i}_czech"), eval(f"data_test_{i}_czech"))
        normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

        training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_czech, y_train_czech, normalized_train_X_german,
                                                    y_train_german, normalized_train_X_italian, y_train_italian, normalized_train_X_nls, y_train_nls)

        y_test_italian = y_test_italian.tolist()

        model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
        model.fit(training_data, training_labels)
        grid_predictions = model.predict(normalized_test_X_italian)
        grid_test_scores = model.predict_scores_list(normalized_test_X_italian)

        predictions = predictions + grid_predictions
        truth = truth + y_test_italian
        test_scores += grid_test_scores[:,0].tolist()

    # report
    print('Italian:')
    print(classification_report(truth, predictions, output_dict=False))
    print(confusion_matrix(truth, predictions))

    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(truth,test_scores))
    print('----------')

    # Test - German -----------------------
    predictions = []
    truth = []
    test_scores = []
    for i in range(1, 11):

        normalized_train_X_nls, normalized_test_X_nls, y_train_nls, y_test_nls = normalize(eval(f"data_train_{i}_nls"), eval(f"data_test_{i}_nls"))
        normalized_train_X_italian, normalized_test_X_italian, y_train_italian, y_test_italian = normalize(eval(f"data_train_{i}_italian"), eval(f"data_test_{i}_italian"))
        normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
        normalized_train_X_czech, normalized_test_X_czech, y_train_czech, y_test_czech = normalize(eval(f"data_train_{i}_czech"), eval(f"data_test_{i}_czech"))
        normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

        training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_czech, y_train_czech, normalized_train_X_german,
                                                    y_train_german, normalized_train_X_italian, y_train_italian, normalized_train_X_nls, y_train_nls)

        y_test_german = y_test_german.tolist()

        model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
        model.fit(training_data, training_labels)
        grid_predictions = model.predict(normalized_test_X_german)
        grid_test_scores = model.predict_scores_list(normalized_test_X_german)

        predictions = predictions + grid_predictions
        truth = truth + y_test_german
        test_scores += grid_test_scores[:,0].tolist()

    # report
    print('German:')
    print(classification_report(truth, predictions, output_dict=False))
    print(confusion_matrix(truth, predictions))

    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(truth,test_scores))
    print('----------')


    # Test - GITA -----------------------
    predictions = []
    truth = []
    test_scores = []
    for i in range(1, 11):

        normalized_train_X_nls, normalized_test_X_nls, y_train_nls, y_test_nls = normalize(eval(f"data_train_{i}_nls"), eval(f"data_test_{i}_nls"))
        normalized_train_X_italian, normalized_test_X_italian, y_train_italian, y_test_italian = normalize(eval(f"data_train_{i}_italian"), eval(f"data_test_{i}_italian"))
        normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
        normalized_train_X_czech, normalized_test_X_czech, y_train_czech, y_test_czech = normalize(eval(f"data_train_{i}_czech"), eval(f"data_test_{i}_czech"))
        normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

        training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_czech, y_train_czech, normalized_train_X_german,
                                                    y_train_german, normalized_train_X_italian, y_train_italian, normalized_train_X_nls, y_train_nls)

        y_test_colombian = y_test_colombian.tolist()

        model = PCA_PLDA_EER_Classifier(PCA_n=best_param,normalize=0)
        model.fit(training_data, training_labels)
        grid_predictions = model.predict(normalized_test_X_colombian)
        grid_test_scores = model.predict_scores_list(normalized_test_X_colombian)

        predictions = predictions + grid_predictions
        truth = truth + y_test_colombian
        test_scores += grid_test_scores[:,0].tolist()

    # report
    print('GITA:')
    print(classification_report(truth, predictions, output_dict=False))
    print(confusion_matrix(truth, predictions))

    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)
    print('specificity')
    print(specificity)
    print('sensitivity')
    print(sensitivity)
    print('ROC_AUC')
    print(roc_auc_score(truth,test_scores))
    print('----------')
    
# print()
# print('PD, HC')
# print((arrayOfSpeaker_pd.shape[0],arrayOfSpeaker_cn.shape[0]))
# print()