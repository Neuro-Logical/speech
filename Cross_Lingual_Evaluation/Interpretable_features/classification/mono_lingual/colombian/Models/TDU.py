BASE = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
SVM_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/GITA/TDU/SVM/'
KNN_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/GITA/TDU/KNN/'
RF_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/GITA/TDU/RF/'
XGBOOST_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/GITA/TDU/XG/'
BAGGING_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/GITA/TDU/BAGG/'

import sys
sys.path.append("/export/b15/afavaro/git_code_version/speech")
from Cross_Lingual_Evaluation.Interpretable_features.classification.mono_lingual.Data_Prep_TDU import *
from Cross_Lingual_Evaluation.Interpretable_features.classification.mono_lingual.Utils import *
import numpy as np
import random
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
random.seed(10)

spain = gita_prep(os.path.join(BASE, "GITA/total_data_frame_novel_task_combined_ling_tot.csv"))
gr = spain.groupby('labels')
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
    data_i = spain[spain["names"].isin(i)]
    data_i = data_i.drop(columns=['names', 'AudioFile', 'task'])
    folds.append((data_i).to_numpy())

data_train_1, data_test_1, data_train_2, data_test_2, data_train_3, data_test_3, data_train_4, data_test_4, \
data_train_5, data_test_5,  data_train_6, data_test_6, data_train_7, data_test_7 , data_train_8, data_test_8, \
data_train_9, data_test_9, data_train_10, data_test_10 = create_split_train_test(folds)

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
    model = SVC(C=10, gamma=0.001, kernel='sigmoid')
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)
    f_1 = report['1']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(SVM_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(SVM_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # KNeighborsClassifier
    model = KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='uniform')
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)
    f_1 = report['1']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(KNN_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(KNN_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # RandomForestClassifier
    model = RandomForestClassifier(max_features='log2', n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)
    f_1 = report['1']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(RF_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(RF_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(learning_rate=0.001, max_depth=3, n_estimators=1000, subsample=0.7)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)
    f_1 = report['1']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(XGBOOST_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(XGBOOST_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # BaggingClassifier
    model = BaggingClassifier(max_samples=0.5, n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(y_test, grid_predictions, output_dict=False))
    report = classification_report(y_test, grid_predictions, output_dict=True)
    f_1 = report['1']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(BAGGING_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(BAGGING_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

