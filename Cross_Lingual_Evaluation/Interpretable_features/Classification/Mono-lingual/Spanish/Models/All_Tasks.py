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
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as pl
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils import shuffle
import random
import os

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

# In[59]:


from sklearn.inspection import plot_partial_dependence

# In[25]:

random.seed(10)


from sklearn.inspection import plot_partial_dependence

# In[25]:

#print("ciao")

spain = pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv")
spain = spain.dropna()
#spain = spain.rename(columns={"id": "AudioFile"})
#spain = spain.iloc[:, 15:]
spain['labels'] = [elem.split("_")[0] for elem in spain['AudioFile'].tolist()]
spain['task'] = [elem.split("_")[1] for elem in spain['AudioFile'].tolist()]
spain['id'] = [elem.split("_")[2] for elem in spain['AudioFile'].tolist()]
print(spain['id'].tolist()[0])

#spain=pd.read_csv("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments_CTP_events.csv")
#spain = spain.rename(columns={"id": "AudioFile"})
#spain = spain.iloc[:, 16:]


#spain = spain[spain['task'].isin(task)]
spain=spain.drop(columns=[  'Unnamed: 0', 'AudioFile', 'task'])


lab = []
for m in spain['labels'].tolist():
    if m == 'PD':
        lab.append(1)
    if m == 'HC':
        lab.append(0)
spain['labels'] = lab
spain

# In[64]:


gr = spain.groupby('labels')

ctrl_ = gr.get_group(0)

pd_ = gr.get_group(1)

# In[65]:


# spain.columns.tolist()


# In[35]:


arrayOfSpeaker_cn = ctrl_['id'].unique()
random.shuffle(arrayOfSpeaker_cn)

arrayOfSpeaker_pd = pd_['id'].unique()
random.shuffle(arrayOfSpeaker_pd)


# In[36]:


def get_n_folds(arrayOfSpeaker):
    data = list(arrayOfSpeaker)  # list(range(len(arrayOfSpeaker)))
    num_of_folds = 10
    n_folds = []
    for i in range(num_of_folds):
        n_folds.append(data[int(i * len(data) / num_of_folds):int((i + 1) * len(data) / num_of_folds)])
    return n_folds


# In[37]:


cn_sps = get_n_folds(arrayOfSpeaker_cn)
pd_sps = get_n_folds(arrayOfSpeaker_pd)
# pd_sps


# In[38]:


data = []
for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
    data.append(cn_sp + pd_sp)
n_folds = sorted(data, key=len, reverse=True)
n_folds

# In[39]:


folds = []
for i in n_folds:
    data_i = spain[spain["id"].isin(i)]
    data_i = data_i.drop(columns=['id'])
    folds.append((data_i).to_numpy())

# In[40]:


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


# In[41]:


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


# In[156]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# define dataset
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


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

    model = SVC(C=50, gamma=0.001, kernel='rbf')
   # model = SVC(C=10, gamma=0.001, kernel='rbf')
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
#
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)
#
    SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/SPANISH/All_Tasks/SVM/'
#
    f_1 = report['1']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(SVM, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(SVM, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))
#

    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    # define dataset
    model = KNeighborsClassifier(metric='manhattan', n_neighbors=17, weights='uniform')
    #model = KNeighborsClassifier(metric='manhattan', n_neighbors= 19, weights= 'uniform')

    grid_result = model.fit(X_train, y_train)

    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)

    SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/SPANISH/All_Tasks/KNN/'
    f_1 = report['1']['f1-score']
    acc = report['accuracy']

    with open(os.path.join(SVM, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))

    with open(os.path.join(SVM, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))



    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier



    model = RandomForestClassifier(max_features= 'sqrt', n_estimators= 1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)

    SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/SPANISH/All_Tasks/RF/'
    f_1 = report['1']['f1-score']
    acc = report['accuracy']

    with open(os.path.join(SVM, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))

    with open(os.path.join(SVM, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))



    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(learning_rate=0.001, max_depth=7, n_estimators=1000, subsample=0.5)
    #model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.5)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)

    SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/SPANISH/All_Tasks/XG/'

    f_1 = report['1']['f1-score']
    acc = report['accuracy']

    with open(os.path.join(SVM, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))

    with open(os.path.join(SVM, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import BaggingClassifier

    model = BaggingClassifier(n_estimators=1000, max_samples=0.5)
   # model = BaggingClassifier(n_estimators=100, max_samples=0.5)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)

    SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/SPANISH/All_Tasks/BAGG/'

    f_1 = report['1']['f1-score']
    acc = report['accuracy']


    with open(os.path.join(SVM, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))

    with open(os.path.join(SVM, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

