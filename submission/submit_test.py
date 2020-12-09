import os
import argparse
import unittest
import random
import copy
import pandas as pd
import numpy as np
from unittest.mock import  patch
from config.consts import ARTIFACTS_ROOT
from submission.submission_utils import calc_auc
from submission.submit import main

COL_NAMES=['image_name', 'target', 'labels']
DATA=[['img1',1,1],['img2',0,0],['img3',3,1],['img4',4,0],['img5',5,1]]
DATA_EXACT=[['img1',1,1],['img2',0,0],['img3',1,1],['img4',0,0],['img5',1,1]]

def read_scv_synth(filename):
    #print(filename)

    data=DATA
    random.shuffle(data)
    df=pd.DataFrame(data=data, columns=COL_NAMES)
    return df

def read_scv_synth_rand(filename):

    data=copy.deepcopy(DATA_EXACT)
    for i in range(len(data)):
        data[i][1]+=np.random.uniform(-5,5)
    random.shuffle(data)
    df=pd.DataFrame(data=data, columns=COL_NAMES)

    return df

def does_df_changed(df_list):
    etalon_df=pd.DataFrame(data=DATA,columns=COL_NAMES)
    etalon_df=etalon_df.sort_values(by='image_name')
    for df in df_list:
        df=df.sort_values(by='image_name')
        for key in COL_NAMES:
            if not all(etalon_df[key].values==df[key].values):
                return True
    return False



class SubmitTTATest(unittest.TestCase):

    @patch('submission.submit.os.path.exists')
    @patch('submission.submit.pd')
    @patch('submission.submission_utils.pd.DataFrame.to_csv')
    @patch('submission.parseargs')
    def test_averaging_does_not_change_equal_submissions(self, mock_parse_args, mock_to_csv, mock_read_csv, mock_exists):
        random.seed(0)
        nfolds=3
        args=argparse.Namespace()
        args.folds=nfolds
        args.work_dir=os.path.join(ARTIFACTS_ROOT,'tmp')
        mock_parse_args.parse_args.return_value=args

        mock_to_csv.return_value=True

        mock_read_csv.read_csv.side_effect = read_scv_synth
        mock_exists.return_value=True

        avg_test_subms, _ = main()

        self.assertEqual(mock_read_csv.read_csv.call_count,nfolds*2*2*2)

        self.assertTrue(not does_df_changed(avg_test_subms))

    @patch('submission.submit.os.path.exists')
    @patch('submission.submit.pd')
    @patch('submission.submission_utils.pd.DataFrame.to_csv')
    @patch('submission.parseargs')
    def test_averaging_improves_auc(self, mock_parse_args, mock_to_csv, mock_read_csv, mock_exists):
        random.seed(0)
        nfolds=3
        args=argparse.Namespace()
        args.folds=nfolds
        args.work_dir=os.path.join(ARTIFACTS_ROOT,'tmp')
        mock_parse_args.parse_args.return_value=args

        mock_to_csv.return_value=True

        mock_read_csv.read_csv.side_effect = read_scv_synth_rand
        mock_exists.return_value=True

        avg_test_subms, mean_auc_val = main()

        avg_test_auc=calc_auc(avg_test_subms[-1])
        self.assertGreater(avg_test_auc,mean_auc_val)





if __name__=="__main__":
    unittest.main()
