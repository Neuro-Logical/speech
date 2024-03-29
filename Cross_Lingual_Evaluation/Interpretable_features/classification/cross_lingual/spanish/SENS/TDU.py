BASE_DIR = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
SPEC_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/SPANISH/TDU/SPEC/'
SENS_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/SPANISH/TDU/SENS/'

sys.path.append("/export/b15/afavaro/git_code_version/speech")
from Cross_Lingual_Evaluation.Interpretable_features.classification.cross_lingual.Data_Prep_TDU import *
from Cross_Lingual_Evaluation.Interpretable_features.classification.cross_lingual.Utils_TDU import *
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
np.random.seed(20)

colombian, colombian_cols = gita_prep(os.path.join(BASE_DIR, "GITA/total_data_frame_novel_task_combined_ling_tot.csv"))
spain, spain_cols = neurovoz_prep(os.path.join(BASE_DIR, "NEUROVOZ/tot_data_experiments.csv"))
german, german_cols = german_prep(os.path.join(BASE_DIR, "GERMAN/final_data_frame_with_intensity.csv"))
italian, italian_cols = italian_prep(os.path.join(BASE_DIR, "ITALIAN_PD/tot_experiments_ling_fin.csv"))

one_inter = IntersecOftwo(german_cols, italian_cols)
list_to_keep = IntersecOfSets(one_inter, colombian_cols,spain_cols)

colombian = colombian[colombian.columns.intersection(list_to_keep)]
german = german[german.columns.intersection(list_to_keep)]
italian = italian[italian.columns.intersection(list_to_keep)]
spain = spain[spain.columns.intersection(list_to_keep)]

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
with open(os.path.join(SPEC_OUT_PATH, f"SVM_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
with open(os.path.join(SENS_OUT_PATH, f"SVM_sens.txt"), 'w') as f:
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
with open(os.path.join(SPEC_OUT_PATH, f"KNN_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
with open(os.path.join(SENS_OUT_PATH, f"KNN_sens.txt"), 'w') as f:
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
with open(os.path.join(SPEC_OUT_PATH, f"RF_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
with open(os.path.join(SENS_OUT_PATH, f"RF_sens.txt"), 'w') as f:
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
with open(os.path.join(SPEC_OUT_PATH, f"XG_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
with open(os.path.join(SENS_OUT_PATH, f"XG_sens.txt"), 'w') as f:
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
with open(os.path.join(SPEC_OUT_PATH, f"BAGG_spec.txt"), 'w') as f:
    f.writelines(str(specificity))
with open(os.path.join(SENS_OUT_PATH, f"BAGG_sens.txt"), 'w') as f:
    f.writelines(str(sensitivity))






