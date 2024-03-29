BASE = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
SPEC_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/ENGLISH/RP/SPEC/'
SENS_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_2/ENGLISH/RP/SENS/'

import sys
sys.path.append("/export/b15/afavaro/git_code_version/speech")
from Cross_Lingual_Evaluation.Interpretable_features.classification.mono_lingual.Data_Prep_RP import *
from Cross_Lingual_Evaluation.Interpretable_features.classification.mono_lingual.Utils import *
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

german = german_prep(os.path.join(BASE, "GERMAN/final_data_frame_with_intensity.csv"))
gr = german.groupby('labels')
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
    data_i = german[german["names"].isin(i)]
    data_i = data_i.drop(columns=['names'])
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
    model = SVC(C=1.0, gamma=0.001, kernel='rbf')
   # model = SVC(C=10, gamma= 0.01, kernel= 'rbf')
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    cm = (confusion_matrix(y_test, grid_predictions))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)
    with open(os.path.join(SPEC_OUT_PATH, f"SVM_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    with open(os.path.join(SENS_OUT_PATH, f"SVM_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))

    # KNeighborsClassifier
    model = KNeighborsClassifier(metric='manhattan', n_neighbors=9, weights='distance')
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    cm = (confusion_matrix(y_test, grid_predictions))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)
    with open(os.path.join(SPEC_OUT_PATH, f"KNN_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    with open(os.path.join(SENS_OUT_PATH, f"KNN_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))

    # define dataset
    model = RandomForestClassifier(max_features='log2', n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    cm = (confusion_matrix(y_test, grid_predictions))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)
    with open(os.path.join(SPEC_OUT_PATH, f"RF_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    with open(os.path.join(SENS_OUT_PATH, f"RF_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=1000, subsample=0.5)
    #model = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, n_estimators=1000, subsample=0.5)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    cm = (confusion_matrix(y_test, grid_predictions))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)
    with open(os.path.join(SPEC_OUT_PATH, f"XG_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    with open(os.path.join(SENS_OUT_PATH, f"XG_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))

    # BaggingClassifier
    model = BaggingClassifier(max_samples=0.1, n_estimators=1000)
    #model = BaggingClassifier(max_samples=0.2, n_estimators=1000)
    grid_result = model.fit(X_train, y_train)
    grid_predictions = grid_result.predict(X_test)
    cm = (confusion_matrix(y_test, grid_predictions))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print('spec : ', specificity)
    with open(os.path.join(SPEC_OUT_PATH, f"BAGG_spec_{i}.txt"), 'w') as f:
        f.writelines(str(specificity))
    with open(os.path.join(SENS_OUT_PATH, f"BAGG_sens_{i}.txt"), 'w') as f:
        f.writelines(str(sensitivity))



