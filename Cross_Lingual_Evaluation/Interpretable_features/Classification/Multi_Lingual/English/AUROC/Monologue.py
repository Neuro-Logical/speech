BASE = "/export/b15/afavaro/Frontiers/submission/Statistical_Analysis"

from Cross_Lingual_Evaluation.Interpretable_features.Classification.Multi_Lingual.Data_Prep_Monologue import *
from Cross_Lingual_Evaluation.Interpretable_features.Classification.Multi_Lingual.Utils_monologue import *
import numpy as np
import random
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
random.seed(10)

nls, nls_cols = nls_prep(os.path.join(BASE, "/NLS/total_new_training.csv"))
colombian, colombian_cols = gita_prep(os.path.join(BASE, "/GITA/total_data_frame_novel_task_combined_ling_tot.csv"))
spain, spain_cols = neurovoz_prep(os.path.join(BASE,  "/NEUROVOZ/tot_data_experiments.csv"))
czech, czech_clols = czech_prep(os.path.join(BASE, "/Czech/final_data_experiments_updated.csv"))
german, german_cols = german_prep(os.path.join(BASE, "/GERMAN/final_data_frame_with_intensity.csv"))

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


data_train_1_nls = np.concatenate(nls_folds[:9])
data_test_1_nls = np.concatenate(nls_folds[-1:])

data_train_2_nls = np.concatenate(nls_folds[1:])
data_test_2_nls = np.concatenate(nls_folds[:1])

data_train_3_nls = np.concatenate(nls_folds[2:] + nls_folds[:1])
data_test_3_nls = np.concatenate(nls_folds[1:2])

data_train_4_nls = np.concatenate(nls_folds[3:] + nls_folds[:2])
data_test_4_nls = np.concatenate(nls_folds[2:3])

data_train_5_nls = np.concatenate(nls_folds[4:] + nls_folds[:3])
data_test_5_nls = np.concatenate(nls_folds[3:4])

data_train_6_nls = np.concatenate(nls_folds[5:] + nls_folds[:4])
data_test_6_nls = np.concatenate(nls_folds[4:5])

data_train_7_nls = np.concatenate(nls_folds[6:] + nls_folds[:5])
data_test_7_nls = np.concatenate(nls_folds[5:6])

data_train_8_nls = np.concatenate(nls_folds[7:] + nls_folds[:6])
data_test_8_nls = np.concatenate(nls_folds[6:7])

data_train_9_nls = np.concatenate(nls_folds[8:] + nls_folds[:7])
data_test_9_nls = np.concatenate(nls_folds[7:8])

data_train_10_nls = np.concatenate(nls_folds[9:] + nls_folds[:8])
data_test_10_nls = np.concatenate(nls_folds[8:9])


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

# %% md

## German

# %%

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


czech = preprocess_data_frame(czech)
czech_folds = create_n_folds(czech)


data_train_1_czech = np.concatenate(czech_folds[:9])
data_test_1_czech = np.concatenate(czech_folds[-1:])

data_train_2_czech = np.concatenate(czech_folds[1:])
data_test_2_czech = np.concatenate(czech_folds[:1])

data_train_3_czech = np.concatenate(czech_folds[2:] + czech_folds[:1])
data_test_3_czech = np.concatenate(czech_folds[1:2])

data_train_4_czech = np.concatenate(czech_folds[3:] + czech_folds[:2])
data_test_4_czech = np.concatenate(czech_folds[2:3])

data_train_5_czech = np.concatenate(czech_folds[4:] + czech_folds[:3])
data_test_5_czech = np.concatenate(czech_folds[3:4])

data_train_6_czech = np.concatenate(czech_folds[5:] + czech_folds[:4])
data_test_6_czech = np.concatenate(czech_folds[4:5])

data_train_7_czech = np.concatenate(czech_folds[6:] + czech_folds[:5])
data_test_7_czech = np.concatenate(czech_folds[5:6])

data_train_8_czech = np.concatenate(czech_folds[7:] + czech_folds[:6])
data_test_8_czech = np.concatenate(czech_folds[6:7])

data_train_9_czech = np.concatenate(czech_folds[8:] + czech_folds[:7])
data_test_9_czech = np.concatenate(czech_folds[7:8])

data_train_10_czech = np.concatenate(czech_folds[9:] + czech_folds[:8])
data_test_10_czech = np.concatenate(czech_folds[8:9])


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



for i in range(1, 11):

    print(i)

    normalized_train_X_nls, normalized_test_X_nls, y_train_nls, y_test_nls = normalize(eval(f"data_train_{i}_nls"), eval(f"data_test_{i}_nls"))
    normalized_train_X_spain, normalized_test_X_spain, y_train_spain, y_test_spain = normalize(eval(f"data_train_{i}_spain"), eval(f"data_test_{i}_spain"))
    normalized_train_X_german, normalized_test_X_german, y_train_german, y_test_german = normalize(eval(f"data_train_{i}_german"), eval(f"data_test_{i}_german"))
    normalized_train_X_czech, normalized_test_X_czech, y_train_czech, y_test_czech = normalize(eval(f"data_train_{i}_czech"), eval(f"data_test_{i}_czech"))
    normalized_train_X_colombian, normalized_test_X_colombian, y_train_colombian, y_test_colombian = normalize(eval(f"data_train_{i}_colombian"), eval(f"data_test_{i}_colombian"))

    training_data, training_labels = train_split(normalized_train_X_colombian, y_train_colombian,normalized_train_X_czech, y_train_czech, normalized_train_X_german,
                                                  y_train_german, normalized_train_X_spain, y_train_spain, normalized_train_X_nls, y_train_nls)


    test_data, test_labels  = normalized_test_X_nls, y_test_nls

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(training_data, training_labels)
    model = SelectFromModel(clf, prefit=True, max_features=40)
    X_train = model.transform(training_data)
    cols = model.get_support(indices=True)
    X_test = test_data[:, cols]


    model = SVC(C=1.0, gamma=0.01, kernel='rbf', probability=True)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]

    lr_auc = roc_auc_score(test_labels, grid_predictions)
    print(f"auroc is {lr_auc}")


    SVM = '/export/b15/afavaro/Frontiers/submission/Classification_With_Feats_Selection/Cross_Val_Results_Multi/ENGLISH/SS/AUROC/'

    with open(os.path.join(SVM, f"SVM_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))



    model = KNeighborsClassifier(metric='euclidean', n_neighbors=19, weights='distance')
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]
    lr_auc = roc_auc_score(test_labels, grid_predictions)


    with open(os.path.join(SVM, f"KNN_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))


    # define dataset
    model = RandomForestClassifier(max_features= 'log2', n_estimators= 1000)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)

    grid_predictions = grid_predictions[:, 1]

    lr_auc = roc_auc_score(test_labels, grid_predictions)

    with open(os.path.join(SVM, f"RF_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))

    # XGBOOST


    model = GradientBoostingClassifier(learning_rate=0.01, max_depth=9, n_estimators=1000, subsample=0.5)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]

    lr_auc = roc_auc_score(test_labels, grid_predictions)
    print(f"auroc is {lr_auc}")

    with open(os.path.join(SVM, f"XGBoost_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))



    model = BaggingClassifier(n_estimators=1000, max_samples=0.5)
    grid_result = model.fit(X_train, training_labels)
    grid_predictions = grid_result.predict_proba(X_test)
    grid_predictions = grid_predictions[:, 1]

    lr_auc = roc_auc_score(test_labels, grid_predictions)
    print(f"auroc is {lr_auc}")


    with open(os.path.join(SVM, f"Bagging_AUROC_{i}.txt"), 'w') as f:
        f.writelines(str(lr_auc))
