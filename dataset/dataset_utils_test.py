import os
import tensorflow as tf
import unittest
from config.consts import test_data_path
from config.config import CONFIG
import numpy as np
from copy import copy
from dataset.dataset_utils import get_test_dataset,get_validation_dataset,get_test_dataset_tta,get_validation_dataset_tta


def get_testval_files():
    test_files = list(tf.io.gfile.glob(os.path.join(test_data_path,'128x128/test*.tfrec')))
    val_files = list(tf.io.gfile.glob(os.path.join(test_data_path,'128x128/train*.tfrec')))
    return test_files, val_files


def is_dataset_deterministic(dataset):
    names = []
    labs = []
    for _, label, name in dataset.unbatch():
        names.append(name.numpy().decode('utf-8'))
        labs.append(label.numpy())
    names2 = []
    labs2 = []
    for _, label, name in dataset.unbatch():
        names2.append(name.numpy().decode('utf-8'))
        labs2.append(label.numpy())
    return all(np.array(names)==np.array(names2)) and all(np.array(labs)==np.array(labs2))


class TestDatasetDeterminsm(unittest.TestCase):

    def testdataset_determinism(self):
        test_files,val_files=get_testval_files()
        test_dataset=get_test_dataset(test_files,config=CONFIG)
        self.assertTrue(is_dataset_deterministic(test_dataset))

    def testdataset_tta_determinism(self):
        test_files,val_files=get_testval_files()
        local_config=copy(CONFIG)
        local_config=local_config._replace(ttas=2)
        test_dataset=get_test_dataset_tta(test_files,config=local_config)
        self.assertTrue(is_dataset_deterministic(test_dataset))

    def runTest(self):
        self.testdataset_determinism()
        self.testdataset_tta_determinism()

class ValDatasetDeterminsm(unittest.TestCase):

    def valdataset_determinism(self):
        _,val_files=get_testval_files()
        val_dataset=get_validation_dataset(val_files,config=CONFIG)
        self.assertTrue(is_dataset_deterministic(val_dataset))

    def valdataset_tta_determinism(self):
        _,val_files=get_testval_files()
        local_config=copy(CONFIG)
        local_config=local_config._replace(ttas=2)
        val_dataset=get_validation_dataset_tta(val_files,config=local_config)
        self.assertTrue(is_dataset_deterministic(val_dataset))

    def runTest(self):
        self.valdataset_determinism()
        self.valdataset_tta_determinism()

if __name__=="__main__":
    unittest.TextTestRunner().run(TestDatasetDeterminsm())
    unittest.TextTestRunner().run(ValDatasetDeterminsm())

