BASE = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
SVM_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/TDU/SVM/'
KNN_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/TDU/KNN/'
RF_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/TDU/RF/'
XGBOOST_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/TDU/XG/'
BAGGING_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/TDU/BAGG/'

import sys
sys.path.append("/export/b15/afavaro/git_code_version/speech")
from Cross_Lingual_Evaluation.Interpretable_features.classification.multi_lingual.Data_Prep_TDU import *
from Cross_Lingual_Evaluation.Interpretable_features.classification.multi_lingual.Utils_TDU import *
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

colombian, colombian_cols = gita_prep(os.path.join(BASE, "GITA/total_data_frame_novel_task_combined_ling_tot.csv"))
german, german_cols = german_prep(os.path.join(BASE, "GERMAN/final_data_frame_with_intensity.csv"))
italian, italian_cols = italian_prep(os.path.join(BASE, "ITALIAN_PD/tot_experiments_ling_fin.csv"))
spain, spain_cols = neurovoz_prep(os.path.join(BASE, "NEUROVOZ/tot_data_experiments.csv"))

one_inter = IntersecOftwo(german_cols, italian_cols)
lista_to_keep = IntersecOfSets(one_inter, colombian_cols,spain_cols)

colombian = colombian[colombian.columns.intersection(lista_to_keep)]
german = german[german.columns.intersection(lista_to_keep)]
italian = italian[italian.columns.intersection(lista_to_keep)]
spain = spain[spain.columns.intersection(lista_to_keep)]


spain = preprocess_data_frame(spain)
spain_folds = create_n_folds(spain)

data_train_1_spain, data_test_1_spain, data_train_2_spain, data_test_2_spain, data_train_3_spain, data_test_3_spain, \
data_train_4_spain, data_test_4_spain, data_train_5_spain, data_test_5_spain, data_train_6_spain, data_test_6_spain, \
data_train_7_spain, data_test_7_spain, data_train_8_spain, data_test_8_spain, data_train_9_spain, data_test_9_spain, \
data_train_10_spain, data_test_10_spain = create_split_train_test(spain_folds)

german = preprocess_data_frame(german)
german_folds = create_n_folds(german)

data_train_1_german, data_test_1_german, data_train_2_german, data_test_2_german, \
data_train_3_german, data_test_3_german, data_train_4_german, data_test_4_german , \
data_train_5_german, data_test_5_german, data_train_6_german, data_test_6_german, \
data_train_7_german, data_test_7_german, data_train_8_german, data_test_8_german, \
data_train_9_german, data_test_9_german, data_train_10_german, \
data_test_10_german = create_split_train_test(german_folds)

colombian = preprocess_data_frame(colombian)
colombian_folds = create_n_folds(colombian)

data_train_1_colombian, data_test_1_colombian, data_train_2_colombian, \
data_test_2_colombian, data_train_3_colombian, data_test_3_colombian , \
data_train_4_colombian, data_test_4_colombian , data_train_5_colombian, \
data_test_5_colombian, data_train_6_colombian, data_test_6_colombian, \
data_train_7_colombian, data_test_7_colombian, data_train_8_colombian, \
data_test_8_colombian, data_train_9_colombian, data_test_9_colombian, \
data_train_10_colombian, data_test_10_colombian = create_split_train_test(colombian_folds)

# italian
italian = preprocess_data_frame(italian)
italian_folds = create_n_folds(italian)

data_train_1_italian, data_test_1_italian, data_train_2_italian, data_test_2_italian, \
data_train_3_italian, data_test_3_italian, data_train_4_italian, data_test_4_italian, data_train_5_italian,\
data_test_5_italian, data_train_6_italian, data_test_6_italian,  data_train_7_italian, data_test_7_italian, \
data_train_8_italian, data_test_8_italian, data_train_9_italian, data_test_9_italian, data_train_10_italian, \
data_test_10_italian = create_split_train_test(italian_folds)

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

    training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian, normalized_train_X_spain, y_train_spain, normalized_train_X_german,
                                                 y_train_german, normalized_train_X_italian, y_train_italian)

    test_data, test_labels = normalized_test_X_spain, y_test_spain

    clf = ExtraTreesClassifier(n_estimators=30)
    clf = clf.fit(training_data, training_labels)
    model = SelectFromModel(clf, prefit=True, max_features=35)
    X_train = model.transform(training_data)
    cols = model.get_support(indices=True)
    X_test = test_data[:, cols]
    model = SVC(C=1.0, gamma=0.01, kernel='rbf')
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(test_labels, grid_predictions, output_dict=False))
    report = classification_report(test_labels, grid_predictions, output_dict=True)
    print(report)
    f_1 = report['1.0']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(SVM_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(SVM_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # KNeighborsClassifier
    model = KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights='uniform')
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(test_labels, grid_predictions, output_dict=False))
    report = classification_report(test_labels, grid_predictions, output_dict=True)
    f_1 = report['1.0']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(KNN_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(KNN_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))
    # RandomForestClassifier
    model = RandomForestClassifier(max_features= 'log2', n_estimators= 1000)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(test_labels, grid_predictions, output_dict=False))
    report = classification_report(test_labels, grid_predictions, output_dict=True)
    f_1 = report['1.0']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(SVM, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(SVM, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=1000, subsample=0.5)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(test_labels, grid_predictions, output_dict=False))
    report = classification_report(test_labels, grid_predictions, output_dict=True)
    f_1 = report['1.0']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(RF_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(RF_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    #BaggingClassifier
    model = BaggingClassifier(n_estimators=1000, max_samples=0.5)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(test_labels, grid_predictions, output_dict=False))
    report = classification_report(test_labels, grid_predictions, output_dict=True)
    f_1 = report['1.0']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(BAGGING_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(BAGGING_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))



