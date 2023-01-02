from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append("/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Multi_Lingual_PD/")
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
from Cross_Validation.Utils import *
from Cross_Validation.Data_Prep_RP import *
from sklearn.metrics import roc_auc_score
random.seed(10)

test = nls_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/Data_frame_RP.csv")
gr = test.groupby('label')
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
    data_i = test[test["id"].isin(i)]
    data_i = data_i.drop(columns=['AudioFile', 'id'])
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



for i in range(1, 11):

    print(i)

    normalized_train_X, normalized_test_X, y_train, y_test = normalize(eval(f"data_train_{i}"), eval(f"data_test_{i}"))

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(normalized_train_X, y_train)
    model = SelectFromModel(clf, prefit=True, max_features=30)
    X_train = model.transform(normalized_train_X)
    cols = model.get_support(indices=True)
    X_test = normalized_test_X[:, cols]

    # SVC
    model = SVC(C=50, gamma= 0.001, kernel= 'sigmoid', probability=True)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)
    print(f"auroc is {lr_auc}")
    SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/ENGLISH/RP/AUROC/'

    with open(os.path.join(SVM, f"SVM_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # KNeighborsClassifier
    model = KNeighborsClassifier(metric='euclidean', n_neighbors=15, weights='uniform')
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)

    with open(os.path.join(SVM, f"KNN_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # RandomForestClassifier
    model = RandomForestClassifier(max_features='log2', n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)

    with open(os.path.join(SVM, f"RF_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.5)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)
    print(f"auroc is {lr_auc}")

    with open(os.path.join(SVM, f"XGBoost_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # BaggingClassifier
    model = BaggingClassifier(max_samples=0.2, n_estimators=100)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)
    print(f"auroc is {lr_auc}")

    with open(os.path.join(SVM, f"Bagging_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))
