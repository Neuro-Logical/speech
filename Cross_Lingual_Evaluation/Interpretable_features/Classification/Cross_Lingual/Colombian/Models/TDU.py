import sys
sys.path.append("/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Lingual/")
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
from Cross_Linguall.Utils_TDU import *
from Cross_Linguall.Data_Prep_TDU import *
np.random.seed(20)

colombian, colombian_cols = gita_prep(
    "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GITA/total_data_frame_novel_task_combined_ling_tot.csv")
spain, spain_cols = neurovoz_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/NEUROVOZ/tot_data_experiments.csv")
german, german_cols = german_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/GERMAN/final_data_frame_with_intensity.csv")
italian, italian_cols = italian_prep("/export/b15/afavaro/Frontiers/submission/Statistical_Analysis/ITALIAN_PD/tot_experiments_ling_fin.csv")

one_inter = IntersecOftwo(german_cols, italian_cols)
lista_to_keep = IntersecOfSets(one_inter, colombian_cols, spain_cols)

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

normalized_train_X_german, y_train_german, mean_german, std_german = normalize(german)
normalized_train_X_italian, y_train_italian, mean_italian, std_italian = normalize(italian)
normalized_train_X_spain, y_train_spain, mean_spain, std_spain = normalize(spain)

means = np.mean(np.stack([mean_spain, mean_german, mean_italian], axis=1), axis=1)
stds = np.mean(np.stack([std_spain, std_german, std_italian], axis=1), axis=1)

normalized_train_X_colombian, y_train_colombian = normalize_test(colombian, means, stds)

training_data, training_labels = train_split(normalized_train_X_spain, y_train_spain, normalized_train_X_italian,
                                             y_train_italian,
                                             normalized_train_X_german, y_train_german)

test_data, test_labels = test_split(normalized_train_X_colombian, y_train_colombian)

clf = ExtraTreesClassifier(n_estimators=30)
clf = clf.fit(training_data, training_labels)
model = SelectFromModel(clf, prefit=True, max_features=40)
X_train = model.transform(training_data)
cols = model.get_support(indices=True)
X_test = test_data[:, cols]

# SVC
model = SVC(C=1.0, gamma=0.01, kernel='rbf')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
print(report)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GITA/TDU/SVM/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))
with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))

#model = KNeighborsClassifier()
model = KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights='uniform')
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GITA/TDU/KNN/'
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
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GITA/TDU/RF/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))

with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))

#model = GradientBoostingClassifier()
model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.5)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GITA/TDU/XG/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))

with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))

#model = BaggingClassifier()
model = BaggingClassifier(max_samples=0.5, n_estimators=1000)
grid_result = model.fit(X_train, training_labels)
grid_predictions = grid_result.predict(X_test)
print(classification_report(test_labels, grid_predictions, output_dict=False))
report = classification_report(test_labels, grid_predictions, output_dict=True)
SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Cross_Mean1/GITA/TDU/BAGG/'
f_1 = report['1.0']['f1-score']
acc = report['accuracy']

with open(os.path.join(SVM, f"all_f1.txt"), 'w') as f:
    f.writelines(str(f_1))

with open(os.path.join(SVM, f"all_acc.txt"), 'w') as f:
    f.writelines(str(acc))
