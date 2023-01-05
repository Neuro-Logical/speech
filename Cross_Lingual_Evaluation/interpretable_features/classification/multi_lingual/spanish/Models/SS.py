BASE = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"
SVM_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/SS/SVM/'
KNN_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/SS/KNN/'
RF_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/SS/RF/'
XGBOOST_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/SS/XG/'
BAGGING_OUT_PATH = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/SPANISH/SS/BAGG/'

import sys
sys.path.append("/export/b15/afavaro/git_code_version/speech")
from Cross_Lingual_Evaluation.interpretable_features.classification.multi_lingual.Data_Prep_SS import *
from Cross_Lingual_Evaluation.interpretable_features.classification.multi_lingual.Utils_SS import *
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

nls, nls_cols = nls_prep(os.path.join(BASE, "NLS/total_new_training.csv"))
colombian, colombian_cols = gita_prep(os.path.join(BASE, "GITA/total_data_frame_novel_task_combined_ling_tot.csv"))
spain, spain_cols = neurovoz_prep(os.path.join(BASE,  "NEUROVOZ/tot_data_experiments.csv"))
czech, czech_clols = czech_prep(os.path.join(BASE, "Czech/final_data_experiments_updated.csv"))
german, german_cols = german_prep(os.path.join(BASE, "GERMAN/final_data_frame_with_intensity.csv"))

one_inter = IntersecOftwo(german_cols, nls_cols)
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
nls_folds = create_n_folds(nls)

data_train_1_nls, data_test_1_nls, data_train_2_nls, data_test_2_nls, \
data_train_3_nls, data_test_3_nls, data_train_4_nls, data_test_4_nls , data_train_5_nls, \
data_test_5_nls, data_train_6_nls, data_test_6_nls, data_train_7_nls,data_test_7_nls , \
data_train_8_nls, data_test_8_nls , data_train_9_nls, data_test_9_nls , data_train_10_nl, \
data_test_10_nls = create_split_train_test(nls_folds)

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

czech = preprocess_data_frame(czech)
czech_folds = create_n_folds(czech)

data_train_1_czech, data_test_1_czech, data_train_2_czech, data_test_2_czech, \
data_train_3_czech, data_test_3_czech,  data_train_4_czech, data_test_4_czech, \
data_train_5_czech, data_test_5_czech,data_train_6_czech, data_test_6_czech , \
data_train_7_czech, data_test_7_czech, data_train_8_czech, data_test_8_czech , \
data_train_9_czech, data_test_9_czech, data_train_10_czech,\
data_test_10_czech  = create_split_train_test(czech_folds)

colombian = preprocess_data_frame(colombian)
colombian_folds = create_n_folds(colombian)

data_train_1_colombian, data_test_1_colombian, data_train_2_colombian, \
data_test_2_colombian, data_train_3_colombian, data_test_3_colombian , \
data_train_4_colombian, data_test_4_colombian , data_train_5_colombian, \
data_test_5_colombian, data_train_6_colombian, data_test_6_colombian, \
data_train_7_colombian, data_test_7_colombian, data_train_8_colombian, \
data_test_8_colombian, data_train_9_colombian, data_test_9_colombian, \
data_train_10_colombian, data_test_10_colombian = create_split_train_test(colombian_folds)

for i in range(1, 11):

    print(i)

    normalized_train_X_nls, normalized_test_X_nls, y_train_nls, y_test_nls = normalize(eval(f"data_train_{i}_nls"), eval(f"data_test_{i}_nls"))
    normalized_train_X_spain, normalized_test_X_spain, y_train_spain, y_test_spain = normalize(eval(f"data_train_{i}_spain"), eval(f"data_test_{i}_spain"))
    normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
    normalized_train_X_czech, normalized_test_X_czech, y_train_czech, y_test_czech = normalize(eval(f"data_train_{i}_czech"), eval(f"data_test_{i}_czech"))
    normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

    training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian,normalized_train_X_czech, y_train_czech, normalized_train_X_german,
                                                  y_train_german, normalized_train_X_spain, y_train_spain, normalized_train_X_nls, y_train_nls)


    test_data, test_labels  = normalized_test_X_spain, y_test_spain

    clf = ExtraTreesClassifier(n_estimators=50)
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
    f_1 = report['1.0']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(SVM_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(SVM_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # KNeighborsClassifier
    model = KNeighborsClassifier(metric='euclidean', n_neighbors=19, weights='distance')
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
    with open(os.path.join(RF_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(RF_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # GradientBoostingClassifier
    model = GradientBoostingClassifier(learning_rate=0.01, max_depth=9, n_estimators=1000, subsample=0.5)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict(X_test)
    print(classification_report(test_labels, grid_predictions, output_dict=False))
    report = classification_report(test_labels, grid_predictions, output_dict=True)
    f_1 = report['1.0']['f1-score']
    acc = report['accuracy']
    with open(os.path.join(XGBOOST_OUT_PATH, f"all_f1_{i}.txt"), 'w') as f:
        f.writelines(str(f_1))
    with open(os.path.join(XGBOOST_OUT_PATH, f"all_acc_{i}.txt"), 'w') as f:
        f.writelines(str(acc))

    # BaggingClassifier
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

