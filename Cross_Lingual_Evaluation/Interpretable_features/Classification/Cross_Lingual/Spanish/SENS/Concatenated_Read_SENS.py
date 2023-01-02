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
from Cross_Validation_Cross_Mean.Utils_TDU import *
from Cross_Validation_Cross_Mean.Data_Prep_TDU import *
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
np.random.seed(20)

colombian, colombian_cols = gita_prep(
    "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined_ling_tot.csv")
spain, spain_cols = neurovoz_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv")
german, german_cols = german_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/final_data_frame_with_intensity.csv")
italian, italian_cols = italian_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/tot_experiments_ling_fin.csv")

one_inter = IntersecOftwo(german_cols, italian_cols)
lista_to_keep = IntersecOfSets(one_inter, colombian_cols,spain_cols)

colombian = colombian[colombian.columns.intersection(lista_to_keep)]
german = german[german.columns.intersection(lista_to_keep)]
italian = italian[italian.columns.intersection(lista_to_keep)]
spain = spain[spain.columns.intersection(lista_to_keep)]

colombian = colombian.reindex(sorted(colombian.columns), axis=1)
german = german.reindex(sorted(german.columns), axis=1)
spain = spain.reindex(sorted(spain.columns), axis=1)
italian = italian.reindex(sorted(italian.columns), axis=1)

german = preprocess_data_frame(german)
italian = preprocess_data_frame(italian)
colombian = preprocess_data_frame(colombian)
spain = preprocess_data_frame(spain)

normalized_train_X_colombian, y_train_colombian, mean_colombian, std_colombian = normalize(colombian)
normalized_train_X_italian, y_train_italian, mean_italian, std_italian = normalize(italian)
normalized_train_X_german, y_train_german, mean_german, std_german = normalize(german)

means = np.mean(np.stack([mean_german, mean_colombian, mean_italian], axis=1), axis=1)
stds = np.mean(np.stack([std_german, std_colombian, std_italian], axis=1), axis=1)

normalized_train_X_spain, y_train_spain = normalize_test(spain, means, stds)

training_data, training_labels = train_split(normalized_train_X_german, y_train_german, normalized_train_X_italian,
                                             y_train_italian,
                                             normalized_train_X_colombian, y_train_colombian)

test_data, test_labels = test_split(normalized_train_X_spain, y_train_spain)

clf = ExtraTreesClassifier(n_estimators=30)
clf = clf.fit(training_data, training_labels)
model = SelectFromModel(clf, prefit=True, max_features=40)
X_train = model.transform(training_data)
cols = model.get_support(indices=True)
X_test = test_data[:, cols]

#model = SVC(probability=True)
model = SVC(C=1.0, gamma=0.01, kernel='rbf')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
cm = (confusion_matrix(test_labels, grid_predictions))
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity)
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('spec : ', specificity)
SPEC = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/SPANISH/TDU/SPEC/'
SENS = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/SPANISH/TDU/SENS/'

with open(os.path.join(SPEC, f"SVM_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
#
with open(os.path.join(SENS, f"SVM_sens.txt"), 'w') as f:
    f.writelines(str(sensitivity))

# KNeighborsClassifier
model = KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights='uniform')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
cm = (confusion_matrix(test_labels, grid_predictions))
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity)
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('spec : ', specificity)

with open(os.path.join(SPEC, f"KNN_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
#
with open(os.path.join(SENS, f"KNN_sens.txt"), 'w') as f:
    f.writelines(str(sensitivity))

# RandomForestClassifier
model = RandomForestClassifier(max_features='log2', n_estimators=100)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
cm = (confusion_matrix(test_labels, grid_predictions))
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity)
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('spec : ', specificity)

with open(os.path.join(SPEC, f"RF_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
#
with open(os.path.join(SENS, f"RF_sens.txt"), 'w') as f:
    f.writelines(str(sensitivity))

#GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.5)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
cm = (confusion_matrix(test_labels, grid_predictions))
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity)
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('spec : ', specificity)

with open(os.path.join(SPEC, f"XG_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
#
with open(os.path.join(SENS, f"XG_sens.txt"), 'w') as f:
    f.writelines(str(sensitivity))

# BaggingClassifier
model = BaggingClassifier(max_samples=0.5, n_estimators=1000)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
cm = (confusion_matrix(test_labels, grid_predictions))
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity)
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('spec : ', specificity)

with open(os.path.join(SPEC, f"BAGG_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
#
with open(os.path.join(SENS, f"BAGG_sens.txt"), 'w') as f:
    f.writelines(str(sensitivity))






