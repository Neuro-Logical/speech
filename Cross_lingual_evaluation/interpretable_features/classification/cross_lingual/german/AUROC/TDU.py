BASE_DIR = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/"
OUT_PATH ='/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GERMAN/TDU/AUROC/'

sys.path.append("/export/b15/afavaro/git_code_version/speech")
from Cross_lingual_evaluation.interpretable_features.classification.cross_lingual.Data_Prep_TDU import *
from Cross_lingual_evaluation.interpretable_features.classification.cross_lingual.Utils_TDU import *
import numpy as np
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
np.random.seed(20)

colombian, colombian_cols = gita_prep(os.path.join(BASE_DIR, "GITA/total_data_frame_novel_task_combined_ling_tot.csv"))
spain, spain_cols = neurovoz_prep(os.path.join(BASE_DIR, "NEUROVOZ/tot_data_experiments.csv"))
german, german_cols = german_prep(os.path.join(BASE_DIR, "GERMAN/final_data_frame_with_intensity.csv"))
italian, italian_cols = italian_prep(os.path.join(BASE_DIR, "ITALIAN_PD/tot_experiments_ling_fin.csv"))

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
normalized_train_X_spain, y_train_spain, mean_spain, std_spain = normalize(spain)

means = np.mean(np.stack([mean_spain, mean_colombian, mean_italian], axis=1), axis=1)
stds = np.mean(np.stack([std_spain, std_colombian, std_italian], axis=1), axis=1)

normalized_train_X_german, y_train_german = normalize_test(german, means, stds)

training_data, training_labels = train_split(normalized_train_X_spain, y_train_spain, normalized_train_X_italian,
                                             y_train_italian,
                                             normalized_train_X_colombian, y_train_colombian)

test_data, test_labels = test_split(normalized_train_X_german, y_train_german)

clf = ExtraTreesClassifier(n_estimators=30)
clf = clf.fit(training_data, training_labels)
model = SelectFromModel(clf, prefit=True, max_features=40)
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
with open(os.path.join(OUT_PATH, f"SVM_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))

# KNeighborsClassifier
model = KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights='uniform')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
with open(os.path.join(OUT_PATH, f"KNN_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))

#model = RandomForestClassifier()
model = RandomForestClassifier(max_features='log2', n_estimators=1000)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
with open(os.path.join(OUT_PATH, f"RF_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))

# XGBOOST
#model = GradientBoostingClassifier()
model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.5)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
print(f"auroc is {lr_auc}")
with open(os.path.join(OUT_PATH, f"XGBoost_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))

#model = BaggingClassifier()
model = BaggingClassifier(max_samples=0.5, n_estimators=1000)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
print(f"auroc is {lr_auc}")
with open(os.path.join(OUT_PATH, f"Bagging_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))