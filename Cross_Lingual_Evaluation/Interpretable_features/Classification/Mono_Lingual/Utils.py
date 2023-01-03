import numpy as np


def get_n_folds(arrayOfSpeaker):
    data = list(arrayOfSpeaker)  # list(range(len(arrayOfSpeaker)))
    num_of_folds = 10
    n_folds = []
    for i in range(num_of_folds):
        n_folds.append(data[int(i * len(data) / num_of_folds):int((i + 1) * len(data) / num_of_folds)])
    return n_folds


def normalize(train_split, test_split):
    train_set = train_split
    test_set = test_split
    feat_train = train_set[:, :-1]
    lab_train = train_set[:, -1:]
    lab_train = lab_train.astype('int')

    feat_test = test_set[:, :-1]
    lab_test = test_set[:, -1:]
    lab_test = lab_test.astype('int')

    X_train, X_test, y_train, y_test = feat_train, feat_test, lab_train, lab_test
    y_test = y_test.ravel()
    y_train = y_train.ravel()
    X_train = X_train.astype('float')
    X_test = X_test.astype('float')
    normalized_test_X = (X_test - X_train.mean(0)) / (X_train.std(0) + 0.01)
    normalized_train_X = (X_train - X_train.mean(0)) / (X_train.std(0) + 0.01)

    return normalized_train_X, normalized_test_X, y_train, y_test


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

    return data_train_1, data_test_1, data_train_2, data_test_2, data_train_3, data_test_3, data_train_4, data_test_4, \
           data_train_5, data_test_5,  data_train_6, data_test_6, data_train_7, data_test_7 , data_train_8, data_test_8, \
           data_train_9, data_test_9, data_train_10, data_test_10

