from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append("/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Multi_Lingual_PD/")
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils import shuffle
import random
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from Cross_Validation_Multi_Lingual.Utils_TDU import *
from Cross_Validation_Multi_Lingual.Data_Prep_TDU import *
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
random.seed(10)

colombian, colombian_cols = gita_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined_ling_tot.csv")
german, german_cols = german_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/final_data_frame_with_intensity.csv")
italian, italian_cols = italian_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/tot_experiments_ling_fin.csv")
spain, spain_cols = neurovoz_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv")

one_inter = IntersecOftwo(german_cols, italian_cols)
lista_to_keep = IntersecOfSets(one_inter, colombian_cols,spain_cols)

colombian = colombian[colombian.columns.intersection(lista_to_keep)]
german = german[german.columns.intersection(lista_to_keep)]
italian = italian[italian.columns.intersection(lista_to_keep)]
spain = spain[spain.columns.intersection(lista_to_keep)]

spain = preprocess_data_frame(spain)
spain_folds = create_n_folds(spain)

data_train_1_spain = np.concatenate(spain_folds[:9])
data_test_1_spain = np.concatenate(spain_folds[-1:])

data_train_2_spain = np.concatenate(spain_folds[1:])
data_test_2_spain = np.concatenate(spain_folds[:1])

data_train_3_spain = np.concatenate(spain_folds[2:] + spain_folds[:1])
data_test_3_spain = np.concatenate(spain_folds[1:2])

data_train_4_spain = np.concatenate(spain_folds[3:] + spain_folds[:2])
data_test_4_spain = np.concatenate(spain_folds[2:3])

data_train_5_spain = np.concatenate(spain_folds[4:] + spain_folds[:3])
data_test_5_spain = np.concatenate(spain_folds[3:4])

data_train_6_spain = np.concatenate(spain_folds[5:] + spain_folds[:4])
data_test_6_spain = np.concatenate(spain_folds[4:5])

data_train_7_spain = np.concatenate(spain_folds[6:] + spain_folds[:5])
data_test_7_spain = np.concatenate(spain_folds[5:6])

data_train_8_spain = np.concatenate(spain_folds[7:] + spain_folds[:6])
data_test_8_spain = np.concatenate(spain_folds[6:7])

data_train_9_spain = np.concatenate(spain_folds[8:] + spain_folds[:7])
data_test_9_spain = np.concatenate(spain_folds[7:8])

data_train_10_spain = np.concatenate(spain_folds[9:] + spain_folds[:8])
data_test_10_spain = np.concatenate(spain_folds[8:9])

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


for i in range(1, 11):

    print(i)

    normalized_train_X_spain, normalized_test_X_spain, y_train_spain, y_test_spain = normalize(
        eval(f"data_train_{i}_spain"), eval(f"data_test_{i}_spain"))
    normalized_train_X_italian, normalized_test_X_italian, y_train_italian, y_test_italian = normalize(
        eval(f"data_train_{i}_italian"), eval(f"data_test_{i}_italian"))
    normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(
        eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
    normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(
        eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

    training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_spain, y_train_spain,
                                                 normalized_train_X_german,
                                                 y_train_german, normalized_train_X_italian, y_train_italian)

    test_data, test_labels = normalized_test_X_spain, y_test_spain

    clf = ExtraTreesClassifier(n_estimators=30)
    clf = clf.fit(training_data, training_labels)
    model = SelectFromModel(clf, prefit=True, max_features=35)
    X_train = model.transform(training_data)
    cols = model.get_support(indices=True)
    X_test = test_data[:, cols]

    # SVC
    model = SVC(C=1.0, gamma=0.01, kernel='rbf', probability=True)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(test_labels, grid_predictions)
    print(f"auroc is {lr_auc}")
    SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/TDU/AUROC'
    with open(os.path.join(SVM, f"SVM_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # KNeighborsClassifier
    model = KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights='uniform')
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(test_labels, grid_predictions)
    with open(os.path.join(SVM, f"KNN_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # RandomForestClassifier
    model = RandomForestClassifier(max_features='log2', n_estimators=1000)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(test_labels, grid_predictions)
    with open(os.path.join(SVM, f"RF_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.5)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(test_labels, grid_predictions)
    print(f"auroc is {lr_auc}")
    with open(os.path.join(SVM, f"XGBoost_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # BaggingClassifier
    model = BaggingClassifier(max_samples=0.5, n_estimators=1000)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(test_labels, grid_predictions)
    print(f"auroc is {lr_auc}")
    with open(os.path.join(SVM, f"Bagging_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))
