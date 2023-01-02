#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

import random
import os
from sklearn.metrics import classification_report, confusion_matrix


random.seed(10)
nls = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/total_new_training.csv")

#nls = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/total_data_experiments.csv")

#nls = nls.iloc[:, 15:]
nls = nls.drop(columns=['Unnamed: 0'])
nls = nls.dropna()

nls = nls.sort_values(by=['names'])
nls['id'] = [m.split("_ses")[0] for m in nls['names'].tolist()]
nls.columns.tolist()


label_seneca = pd.read_excel("/export/b15/afavaro/Book3.xlsx")
label = label_seneca['Label'].tolist()
speak = label_seneca['Participant I.D.'].tolist()
spk2lab_ = {sp: lab for sp, lab in zip(speak, label)}
speak2__ = nls['id'].tolist()
# label


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
#ctrl['age']= ctrl['age'].astype(float)
#ctrl = ctrl[ctrl['age'] > 62]

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

# In[19]:


gr = test.groupby('label')

ctrl_ = gr.get_group(0)

pd_ = gr.get_group(1)




arrayOfSpeaker_cn = ctrl_['id'].unique()
random.shuffle(arrayOfSpeaker_cn)

arrayOfSpeaker_pd = pd_['id'].unique()
random.shuffle(arrayOfSpeaker_pd)



def get_n_folds(arrayOfSpeaker):
    data = list(arrayOfSpeaker)  # list(range(len(arrayOfSpeaker)))
    num_of_folds = 10
    n_folds = []
    for i in range(num_of_folds):
        n_folds.append(data[int(i * len(data) / num_of_folds):int((i + 1) * len(data) / num_of_folds)])
    return n_folds




cn_sps = get_n_folds(arrayOfSpeaker_cn)

pd_sps = get_n_folds(arrayOfSpeaker_pd)


data = []
for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
    data.append(cn_sp + pd_sp)
n_folds = sorted(data, key=len, reverse=True)


# In[30]:


folds = []
for i in n_folds:
    data_i = test[test["id"].isin(i)]
    data_i = data_i.drop(columns=['names', 'id'])
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



def normalize(train_split, test_split):
    train_set = train_split
    test_set = test_split
    # np.random.shuffle(tot)

    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]
    lab_test = lab_test.astype('int')

    # X = StandardScaler().fit_transform(matrix_feat)

    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_test_X = (X_test - X_train.mean(0)) / (X_train.std(0) + 0.01)
    normalized_train_X = (X_train - X_train.mean(0)) / (X_train.std(0) + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test



from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

for i in range(1, 11):

    print(i)

    normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"), eval(f"data_test_{i}"))

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(normalized_train_X, y_train)
    model = SelectFromModel(clf, prefit=True, max_features=30)

    X_train = model.transform(normalized_train_X)
    cols = model.get_support(indices=True)

    X_test = normalized_test_X[:, cols]
    #

    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC


    model = SVC(C=10, gamma=0.001, kernel='rbf')
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)

    cm = (confusion_matrix(y_test, grid_predictions))

    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)
    SPEC = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/ENGLISH/All_Tasks/SPEC/'
    SENS = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/ENGLISH/All_Tasks/SENS/'

    with open(os.path.join(SPEC, f"SVM_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    #
    with open(os.path.join(SENS, f"SVM_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))

    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(metric='minkowski', n_neighbors=19, weights='uniform')
    #model = KNeighborsClassifier(metric='manhattan', n_neighbors= 15, weights= 'uniform')

    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)

    cm = (confusion_matrix(y_test, grid_predictions))

    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)

    with open(os.path.join(SPEC, f"KNN_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    #
    with open(os.path.join(SENS, f"KNN_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))

    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(max_features='sqrt', n_estimators=1000)
    #model = RandomForestClassifier(max_features='sqrt', n_estimators=100)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)

    cm = (confusion_matrix(y_test, grid_predictions))

    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)

    with open(os.path.join(SPEC, f"RF_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    #
    with open(os.path.join(SENS, f"RF_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))

        # XGBOOST

    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, n_estimators=1000, subsample=0.7)
    #model = GradientBoostingClassifier(learning_rate=0.001, max_depth=9, n_estimators=1000, subsample=0.7)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)

    cm = (confusion_matrix(y_test, grid_predictions))

    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)

    with open(os.path.join(SPEC, f"XG_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    #
    with open(os.path.join(SENS, f"XG_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))

        # example of grid searching key hyperparameters for BaggingClassifier

    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import BaggingClassifier


    model = BaggingClassifier(max_samples=0.5, n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    cm = (confusion_matrix(y_test, grid_predictions))

    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)

    with open(os.path.join(SPEC, f"BAGG_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    #
    with open(os.path.join(SENS, f"BAGG_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))





