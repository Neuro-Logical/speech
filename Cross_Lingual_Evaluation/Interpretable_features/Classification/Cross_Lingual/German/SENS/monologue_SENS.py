import sys
sys.path.append("/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Multi_Lingual_PD/")
import numpy as np
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from Cross_Lingual.Utils_monologue import *
from Cross_Lingual.Data_Prep_monologue import *
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
normalized_train_X_nls, y_train_nls, mean_nls, std_nls = normalize(nls)
normalized_train_X_colombian, y_train_colombian, mean_colombian, std_colombian = normalize(colombian)
normalized_train_X_czech, y_train_czech, mean_czech, std_czech = normalize(czech)

means = np.mean(np.stack([mean_spain, mean_nls, mean_colombian, mean_czech], axis=1), axis=1)
stds = np.mean(np.stack([std_spain, std_nls, std_colombian, std_czech], axis=1), axis=1)

normalized_train_X_german, y_train_german = normalize_test(german, means, stds)

training_data, training_labels = train_split(normalized_train_X_spain, y_train_spain, normalized_train_X_czech,
                                             y_train_czech,
                                             normalized_train_X_nls, y_train_nls, normalized_train_X_colombian,
                                             y_train_colombian)

test_data, test_labels = test_split(normalized_train_X_german, y_train_german)

clf = ExtraTreesClassifier(n_estimators=60)
clf = clf.fit(training_data, training_labels)
model = SelectFromModel(clf, prefit=True, max_features=40)
X_train = model.transform(training_data)
cols = model.get_support(indices=True)
X_test = test_data[:, cols]

# SVC
model = SVC(C=10, gamma=0.01, kernel='rbf')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
cm = (confusion_matrix(test_labels, grid_predictions))
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity)
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('spec : ', specificity)
SPEC = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GERMAN/SS/SPEC/'
SENS = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GERMAN/SS/SENS/'

with open(os.path.join(SPEC, f"SVM_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
#
with open(os.path.join(SENS, f"SVM_sens.txt"), 'w') as f:
    f.writelines(str(sensitivity))

# KNeighborsClassifier
model = KNeighborsClassifier(metric='euclidean', n_neighbors=11, weights='distance')
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
model = RandomForestClassifier(max_features='log2', n_estimators=1000)
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

# GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.001, max_depth=9, n_estimators=1000, subsample=0.5)
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
model = BaggingClassifier(max_samples =0.2, n_estimators=1000)
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


