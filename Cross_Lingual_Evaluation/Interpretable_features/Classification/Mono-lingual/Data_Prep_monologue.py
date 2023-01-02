import pandas as pd
import numpy as np

def nls_prep(path_to_dataframe):

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

    return test

def gita_prep(path_to_dataframe):

    spain = pd.read_csv(path_to_dataframe)
    spain = spain.drop(columns=['Unnamed: 0'])
    spain = spain.dropna()
    spain['names'] = [elem.split("_")[1] for elem in spain.AudioFile.tolist()]
    spain['labels'] = [elem.split("_")[0] for elem in spain.AudioFile.tolist()]
    new_lab = []
    for lab in spain['labels'].tolist():
        if lab == "PD":
            new_lab.append(1)
        if lab == "CN":
            new_lab.append(0)
        if lab == "HC":
            new_lab.append(0)
    spain['labels'] = new_lab

    return spain

def neurovoz_prep(path_to_dataframe):

    """Pre-processing function Neurovoz data set."""

    spain = pd.read_csv(path_to_dataframe)
    spain = spain.dropna()
    # spain=pd.read_csv("/export/b15/afavaro/Frontiers/submission/Classification/Spanish/spain_pd_noling.csv")
    spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
    spain['task'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
    spain['id'] = [elem.split("_")[2] for elem in spain['AudioFile'].tolist()]
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

    return spain

def german_prep(path_to_dataframe):

    """Pre-processing function GermanPD data set."""

    spain = pd.read_csv(path_to_dataframe)
    spain = spain.drop(columns=['Unnamed: 0'])
    spain['names'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
    spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
    spain['task'] = [elem.split("_")[-2] for elem in spain['AudioFile'].tolist()]
    task = ['monologue']
    spain = spain[spain['task'].isin(task)]
    spain = spain.drop(columns=['AudioFile', 'task'])
    lab = []
    for m in spain['labels'].tolist():
        if m == "PD":
            lab.append(1)
        if m == 'CN':
            lab.append(0)
    spain['labels'] = lab
    spain = spain.dropna()

    return spain

def italian_prep(path_to_dataframe):

    """Pre-processing function ItalianPVS data set."""

    italian = pd.read_csv(path_to_dataframe)
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
    italian = italian.drop(columns=['Unnamed: 0', 'AudioFile', 'task'])

    return italian

def czech_prep(path_to_dataframe):

    spain = pd.read_csv(path_to_dataframe)
    spain['names'] = [elem.split("_")[1] for elem in spain.AudioFile.tolist()]
    spain['task'] = [elem.split("_")[2] for elem in spain['AudioFile'].tolist()]
    spain['labels'] = [elem.split("_")[0] for elem in spain.AudioFile.tolist()]
    task = ['monologue']
    spain = spain[spain['task'].isin(task)]
    spain = spain.drop(columns=['Unnamed: 0', 'AudioFile', 'task'])
    lab = []
    for m in spain['labels'].tolist():
        if m == "PD":
            lab.append(1)
        if m == 'CN':
            lab.append(0)
    spain['labels'] = lab
    spain = spain.dropna()

    return spain

