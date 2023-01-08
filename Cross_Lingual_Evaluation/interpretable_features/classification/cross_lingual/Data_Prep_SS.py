import random
import numpy as np
import pandas as pd

def nls_prep(path_to_dataframe):

    #nls = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/total_new_training.csv")
    nls = pd.read_csv(path_to_dataframe)
    nls = nls.drop(columns=['Unnamed: 0'])
    nls = nls.dropna()
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

    return nls, nls_cols

def gita_prep(path_to_dataframe):

    colombian = pd.read_csv(path_to_dataframe)
    colombian = colombian.dropna()
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

    return colombian, colombian_cols


def neurovoz_prep(path_to_dataframe):

    spain = pd.read_csv(path_to_dataframe)
    spain = spain.dropna()
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

    return spain, spain_cols


def german_prep(path_to_dataframe):

    german = pd.read_csv(path_to_dataframe)
    german = german.drop(columns=['Unnamed: 0'])
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

    return german, german_cols


def czech_prep(path_to_dataframe):

    czech = pd.read_csv(path_to_dataframe)
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

    return czech, czech_clols