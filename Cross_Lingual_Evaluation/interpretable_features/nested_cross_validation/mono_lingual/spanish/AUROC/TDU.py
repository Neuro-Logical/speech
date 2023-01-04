BASE = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/SPANISH/TDU/AUROC/'

from Cross_Lingual_Evaluation.interpretable_features.nested_cross_validation.mono_lingual.Data_Prep_TDU import *
from Cross_Lingual_Evaluation.interpretable_features.nested_cross_validation.mono_lingual.Utils import *
import random
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
random.seed(10)

spanish = neurovoz_prep(os.path.join(BAE, "/NEUROVOZ/tot_data_experiments.csv"))
gr = spanish.groupby('labels')
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
    data_i = spanish[spanish["id"].isin(i)]
    data_i = data_i.drop(columns=['id'])
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
    model = SVC(C=1.0, gamma=0.01, kernel='sigmoid', probability=True)
   # model = SVC(C=50, gamma= 0.001, kernel= 'sigmoid', probability=True)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)
    print(f"auroc is {lr_auc}")
    with open(os.path.join(OUT_PATH, f"SVM_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # KNeighborsClassifier
    model = KNeighborsClassifier(metric='euclidean', n_neighbors=9, weights='distance')
    #model = KNeighborsClassifier(metric='manhattan', n_neighbors= 19, weights= 'distance')
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)
    with open(os.path.join(OUT_PATH, f"KNN_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # RandomForestClassifier
    model = RandomForestClassifier(max_features='sqrt', n_estimators=1000)
   # model = RandomForestClassifier(max_features='log2', n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)
    with open(os.path.join(OUT_PATH, f"RF_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # XGBOOST
    model = GradientBoostingClassifier(learning_rate=0.001, max_depth=9, n_estimators=1000, subsample=0.7)
    #model = GradientBoostingClassifier(learning_rate=0.01, max_depth=9, n_estimators=100, subsample=0.5)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)
    print(f"auroc is {lr_auc}")
    with open(os.path.join(OUT_PATH, f"XGBoost_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # BaggingClassifier
    model = BaggingClassifier(max_samples=0.5, n_estimators=1000)
    #model = BaggingClassifier(max_samples=0.1, n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(y_test, grid_predictions)
    print(f"auroc is {lr_auc}")
    with open(os.path.join(OUT_PATH, f"Bagging_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))
