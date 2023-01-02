
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
from Cross_Lingual.Utils_monologue import *
from Cross_Lingual.Data_Prep_monologue import *
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
np.random.seed(20)

nls, nls_cols = nls_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NLS/total_new_training.csv")
colombian, colombian_cols = gita_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined_ling_tot.csv")
spain, spain_cols = neurovoz_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv")
german, german_cols = german_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/final_data_frame_with_intensity.csv")
czech, czech_clols = czech_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/Czech/final_data_experiments_updated.csv")

one_inter = IntersecOfSets(german_cols, nls_cols, spain_cols)
lista_to_keep = IntersecOfSets(one_inter, colombian_cols, czech_clols)

nls = nls[nls.columns.intersection(lista_to_keep)]
czech = czech[czech.columns.intersection(lista_to_keep)]
colombian = colombian[colombian.columns.intersection(lista_to_keep)]
german = german[german.columns.intersection(lista_to_keep)]
spain = spain[spain.columns.intersection(lista_to_keep)]

colombian = colombian.reindex(sorted(colombian.columns), axis=1)
german = german.reindex(sorted(german.columns), axis=1)
spain = spain.reindex(sorted(spain.columns), axis=1)
nls = nls.reindex(sorted(nls.columns), axis=1)
czech = czech.reindex(sorted(czech.columns), axis=1)

nls = preprocess_data_frame(nls)
german = preprocess_data_frame(german)
czech = preprocess_data_frame(czech)
spain = preprocess_data_frame(spain)
colombian = preprocess_data_frame(colombian)

normalized_train_X_spain, y_train_spain, mean_spain, std_spain = normalize(spain)
normalized_train_X_german, y_train_german, mean_german, std_german = normalize(german)
normalized_train_X_nls, y_train_nls, mean_nls, std_nls = normalize(nls)
normalized_train_X_colombian, y_train_colombian, mean_colombian, std_colombian = normalize(colombian)
means = np.mean(np.stack([mean_spain, mean_german, mean_nls, mean_colombian], axis=1), axis=1)
stds = np.mean(np.stack([std_spain, std_german, std_nls, std_colombian], axis=1), axis=1)
normalized_train_X_czech, y_train_czech = normalize_test(czech, means, stds)
training_data, training_labels = train_split(normalized_train_X_spain, y_train_spain, normalized_train_X_colombian,
                                             y_train_colombian,
                                             normalized_train_X_german, y_train_german, normalized_train_X_nls,
                                             y_train_nls)
test_data, test_labels = test_split(normalized_train_X_czech, y_train_czech)

clf = ExtraTreesClassifier(n_estimators=60)
clf = clf.fit(training_data, training_labels)
model = SelectFromModel(clf, prefit=True, max_features=45)
X_train = model.transform(training_data)
cols = model.get_support(indices=True)
X_test = test_data[:, cols]

#model = SVC()
model = SVC(C=10, gamma=0.01, kernel='rbf')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
print(report)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/CZECH/SS/SVM/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))
with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))

# KNeighborsClassifier
model = KNeighborsClassifier(metric='euclidean', n_neighbors=11, weights='distance')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/CZECH/SS/KNN/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))

with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))

#model = RandomForestClassifier()
model = RandomForestClassifier(max_features='log2', n_estimators=1000)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/CZECH/SS/RF/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))

with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))

# GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.001, max_depth=9, n_estimators=1000, subsample=0.5)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/CZECH/SS/XG/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))

with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))

#model = BaggingClassifier()
model = BaggingClassifier(n_estimators=1000, max_samples=0.5)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/CZECH/SS/BAGG/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))

with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))
