
import pandas as pd
import numpy as np
import random

random.seed(10)

def normalize(train_split, test_split):

    """Language-wise normalization procedure using the median and the standard deviation of the control group from the training set."""

    train_set = train_split
    test_set = test_split
    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')
    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]
    control_group = train_set[train_set[:, -1] == 0]
    control_group = control_group[:, :-1]  # remove labels from features CNs
    median = np.median(control_group, axis=0)
    std = np.std(control_group, axis=0)
    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_train_X = (X_train - median) / (std + 0.01)
    normalized_test_X = (X_test - median) / (std + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test


def preprocess_data_frame(data_frame):

    """Pre-processing dataframe."""

    nomi = data_frame['id'].tolist()
    lab = data_frame['labels'].tolist()
    data_frame = data_frame.drop(columns=['id', 'labels'])
    data_frame['id'] = nomi
    data_frame['labels'] = lab

    return data_frame


def get_n_folds(arrayOfSpeaker):

    """Create 10 folds (num_of_folds) making sure to not use same the speakers in training and testing."""

    data = list(arrayOfSpeaker)  # list(range(len(arrayOfSpeaker)))
    num_of_folds = 10
    n_folds = []
    for i in range(num_of_folds):
        n_folds.append(data[int(i * len(data) / num_of_folds):int((i + 1) * len(data) / num_of_folds)])
    return n_folds


def create_n_folds(data_frame):

    """Create n-folds making sure to not use same the speakers in training and testing."""

    data = []
    folds = []
    data_grouped = data_frame.groupby('labels')
    ctrl_ = data_grouped.get_group(0)
    pd_ = data_grouped.get_group(1)
    arrayOfSpeaker_cn = ctrl_['id'].unique()
    random.shuffle(arrayOfSpeaker_cn)
    arrayOfSpeaker_pd = pd_['id'].unique()
    random.shuffle(arrayOfSpeaker_pd)
    cn_sps = get_n_folds(arrayOfSpeaker_cn)
    pd_sps = get_n_folds(arrayOfSpeaker_pd)
    for cn_sp, pd_sp in zip(sorted(cn_sps, key=len), sorted(pd_sps, key=len, reverse=True)):
        data.append(cn_sp + pd_sp)
    n_folds = sorted(data, key=len, reverse=True)
    for i in n_folds:
        data_i = data_frame[data_frame["id"].isin(i)]
        data_i = data_i.drop(columns=['id'])
        folds.append((data_i).to_numpy())

    return folds


def IntersecOfSets(arr1, arr2, arr3):

    """Find element in the intersection between three arrays."""

    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)
    set1 = s1.intersection(s2)
    result_set = set1.intersection(s3)
    final_list = list(result_set)

    return final_list


def IntersecOftwo(arr1, arr2):

    """Find element in the intersection between two arrays."""
    s1 = set(arr1)
    s2 = set(arr2)
    set1 = s1.intersection(s2)
    final_list = list(set1)
    return final_list


def train_split(colombian, colombian_lab, czech, czech_lab, spain, spain_lab, german, german_lab,
                english, english_lab):

    """Create matrix and labels for the train set."""

    train_mat_data_point = np.concatenate([colombian, czech, spain, german, english], axis=0)
    train_data_label = np.concatenate([colombian_lab, czech_lab, spain_lab, german_lab, english_lab],
                                      axis=0)

    return train_mat_data_point, train_data_label


def test_split(czech, czech_lab):

    """Create matrix and labels for the test set."""

    train_mat_data_point = np.concatenate([czech], axis=0)
    train_data_label = np.concatenate([czech_lab], axis=0)

    return train_mat_data_point, train_data_label


