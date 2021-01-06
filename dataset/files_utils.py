import re
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import KFold
from config.consts import test_files_num


def count_data_items(filenames):
    if len(filenames)==0:
        return 0
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def get_test_filenames(gs_path_to_dataset_train):
    gcs_pattern = str(gs_path_to_dataset_train).replace('train', 'test')
    test_filenames = tf.io.gfile.glob(gcs_pattern)
    return test_filenames


def get_train_val_filenames(gs_path_to_dataset_train, nfolds):
    filenames = list(tf.io.gfile.glob(str(gs_path_to_dataset_train)))
    random.shuffle(filenames)
    test_val_filenames = filenames[:test_files_num]
    filenames = filenames[test_files_num:]

    filenames = np.array(filenames)
    kf = KFold(n_splits=nfolds, random_state=0, shuffle=True)
    train_filenames_folds = []
    val_filenames_folds = []
    for train_index, test_index in kf.split(filenames):
        train_filenames_folds.append(list(filenames[train_index]))
        val_filenames_folds.append(list(filenames[test_index]))
    return train_filenames_folds, val_filenames_folds, test_val_filenames
