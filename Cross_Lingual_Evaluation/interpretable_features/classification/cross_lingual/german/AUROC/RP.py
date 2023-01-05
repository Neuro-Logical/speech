BASE_DIR = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
OUT_PATH ='/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GERMAN/RP/AUROC/'

sys.path.append("/export/b15/afavaro/git_code_version/speech")
from Cross_Lingual_Evaluation.interpretable_features.classification.cross_lingual.Data_Prep_RP import *
from Cross_Lingual_Evaluation.interpretable_features.classification.cross_lingual.Utils_RP import *
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

nls, nls_cols = nls_prep(os.path.join(BASE_DIR, "/NLS/Data_frame_RP.csv"))
colombian, colombian_cols = gita_prep(os.path.join(BASE_DIR,"/GITA/total_data_frame_novel_task_combined_ling_tot.csv"))
german, german_cols = german_prep(os.path.join(BASE_DIR, "/GERMAN/final_data_frame_with_intensity.csv"))
czech, czech_cols = czech_prep(os.path.join(BASE_DIR, "/czech/final_data_experiments_updated.csv"))
italian, italian_cols = italian_prep(os.path.join(BASE_DIR, "/ITALIAN_PD/RP_data_frame.csv"))

one_inter = IntersecOfSets(german_cols, nls_cols, italian_cols)
lista_to_keep = IntersecOfSets(one_inter, colombian_cols, czech_cols)

nls = nls[nls.columns.intersection(lista_to_keep)]
czech = czech[czech.columns.intersection(lista_to_keep)]
colombian = colombian[colombian.columns.intersection(lista_to_keep)]
german = german[german.columns.intersection(lista_to_keep)]
italian = italian[italian.columns.intersection(lista_to_keep)]

colombian = colombian.reindex(sorted(colombian.columns), axis=1)
german = german.reindex(sorted(german.columns), axis=1)
nls = nls.reindex(sorted(nls.columns), axis=1)
czech = czech.reindex(sorted(czech.columns), axis=1)
italian = italian.reindex(sorted(italian.columns), axis=1)

nls = preprocess_data_frame(nls)
german = preprocess_data_frame(german)
czech = preprocess_data_frame(czech)
italian = preprocess_data_frame(italian)
colombian = preprocess_data_frame(colombian)

normalized_train_X_italian, y_train_italian, mean_italian, std_italian = normalize(italian)
normalized_train_X_nls, y_train_nls, mean_nls, std_nls = normalize(nls)
normalized_train_X_colombian, y_train_colombian, mean_colombian, std_colombian = normalize(colombian)
normalized_train_X_czech, y_train_czech, mean_czech, std_czech = normalize(czech)

means = np.mean(np.stack([mean_italian, mean_nls, mean_colombian, mean_czech], axis=1), axis=1)
stds = np.mean(np.stack([std_italian, std_nls, std_colombian, std_czech], axis=1), axis=1)
normalized_train_X_german, y_train_german = normalize_test(german, means, stds)
training_data, training_labels = train_split(normalized_train_X_italian, y_train_italian, normalized_train_X_czech,
                                             y_train_czech,
                                             normalized_train_X_nls, y_train_nls, normalized_train_X_colombian,
                                             y_train_colombian)

test_data, test_labels = test_split(normalized_train_X_german, y_train_german)

clf = ExtraTreesClassifier(n_estimators=30)
clf = clf.fit(training_data, training_labels)
model = SelectFromModel(clf, prefit=True, max_features=40)
X_train = model.transform(training_data)
cols = model.get_support(indices=True)
X_test = test_data[:, cols]

#model = SVC(probability=True)
model = SVC(C=1.0, gamma= 0.01, kernel= 'rbf', probability=True)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
print(f"auroc is {lr_auc}")
with open(os.path.join(OUT_PATH, f"SVM_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))

# KNeighborsClassifier
model = KNeighborsClassifier(metric='euclidean', n_neighbors=11, weights='distance')
#model = KNeighborsClassifier(metric='manhattan', n_neighbors=9, weights='distance')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
with open(os.path.join(OUT_PATH, f"KNN_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))

# RandomForestClassifier
model = RandomForestClassifier(max_features= 'log2', n_estimators= 1000)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
with open(os.path.join(OUT_PATH, f"RF_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))

# GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.7)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
print(f"auroc is {lr_auc}")
with open(os.path.join(OUT_PATH, f"XGBoost_AUROC_{i}.txt"), 'w') as f:
    f.writelines(str(lr_auc))

# BaggingClassifier
model = BaggingClassifier(max_samples= 0.5, n_estimators=1000)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict_proba(X_test)
grid_predictions = grid_predictions[:, 1]
lr_auc = roc_auc_score(test_labels, grid_predictions)
print(f"auroc is {lr_auc}")
with open(os.path.join(OUT_PATH, f"Bagging_AUROC.txt"), 'w') as f:
    f.writelines(str(lr_auc))
