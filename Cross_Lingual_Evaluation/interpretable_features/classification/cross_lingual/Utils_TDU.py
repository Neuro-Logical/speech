import numpy as np
np.random.seed(20)


def test_split(czech, czech_lab):

    train_mat_data_point = np.concatenate([czech], axis=0)
    train_data_label = np.concatenate([czech_lab], axis=0)

    return train_mat_data_point, train_data_label


def train_split(colombian, colombian_lab, czech, czech_lab, spain, spain_lab):
    train_mat_data_point = np.concatenate([colombian, czech, spain], axis=0)

    train_data_label = np.concatenate([colombian_lab, czech_lab, spain_lab],
                                      axis=0)

    return train_mat_data_point, train_data_label


def concat_train_data(colombian, colombian_lab, czech, czech_lab, spain, spain_lab):

    train_mat_data_point = np.concatenate([colombian, czech, spain], axis=0)
    train_data_label = np.concatenate([colombian_lab, czech_lab, spain_lab], axis=0)

    return train_mat_data_point, train_data_label


def concat_test_data(czech, czech_lab):

    train_mat_data_point = np.concatenate([czech], axis=0)
    train_data_label = np.concatenate([czech_lab], axis=0)

    return train_mat_data_point, train_data_label


def preprocess_data_frame(data_frame):
    # nomi = data_frame['id'].tolist()
    lab = data_frame['labels'].tolist()
    data_frame = data_frame.drop(columns=['labels', 'id'])
    # data_frame['id'] = nomi
    data_frame['labels'] = lab

    return data_frame


def normalize(train_set):

    train_set = train_set.to_numpy()
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1:]
    control_group = train_set[train_set[:, -1] == 0]
    control_group = control_group[:, :-1]  # remove labels from features CNs
    median = np.median(control_group, axis=0)
    std = np.std(control_group, axis=0)
    y_train = y_train.ravel()
    x_train = x_train.astype('float')
    normalized_x_train = (x_train - median) / (std + 0.01)

    return normalized_x_train, y_train, median, std


def normalize_test(test, median, std):

    train_set = test.to_numpy()
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1:]
    y_train = y_train.ravel()
    x_train = x_train.astype('float')
    normalized_x_train = (x_train - median) / (std + 0.01)

    return normalized_x_train, y_train


def IntersecOfSets(arr1, arr2, arr3):

    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)
    set1 = s1.intersection(s2)
    result_set = set1.intersection(s3)
    final_list = list(result_set)

    return final_list


def IntersecOftwo(arr1, arr2):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    set1 = s1.intersection(s2)
    final_list = list(set1)
    return final_list


def create_split_train_test(folds):
    data_train_1 = np.concatenate(folds[:9])
    data_test_1 = np.concatenate(folds[-1:])

    data_train_2 = np.concatenate(folds[1:])
    data_test_2 = np.concatenate(folds[:1])

    data_train_3 = np.concatenate(folds[2:] + folds[:1])
    data_test_3 = np.concatenate(folds[1:2])

    data_train_4 = np.concatenate(folds[3:] + folds[:2])
    data_test_4 = np.concatenate(folds[2:3])

    data_train_5 = np.concatenate(folds[4:] + folds[:3])
    data_test_5 = np.concatenate(folds[3:4])

    data_train_6 = np.concatenate(folds[5:] + folds[:4])
    data_test_6 = np.concatenate(folds[4:5])

    data_train_7 = np.concatenate(folds[6:] + folds[:5])
    data_test_7 = np.concatenate(folds[5:6])

    data_train_8 = np.concatenate(folds[7:] + folds[:6])
    data_test_8 = np.concatenate(folds[6:7])

    data_train_9 = np.concatenate(folds[8:] + folds[:7])
    data_test_9 = np.concatenate(folds[7:8])

    data_train_10 = np.concatenate(folds[9:] + folds[:8])
    data_test_10 = np.concatenate(folds[8:9])

    return data_train_1, data_test_1, data_train_2, data_test_2, \
           data_train_3, data_test_3, data_train_4, data_test_4, \
           data_train_5, data_test_5, data_train_6, data_test_6, data_train_7, data_test_7, data_train_8, \
           data_test_8, data_train_9, data_test_9, data_train_10, data_test_10