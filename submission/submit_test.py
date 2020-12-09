import os
import argparse
import unittest
import pandas as pd
from unittest.mock import  patch
from config.consts import ARTIFACTS_ROOT
from submission.submit import main

def read_scv_synth(filename):
    return pd.DataFrame()

class SubmitTTATest(unittest.TestCase):

    @patch('submission.submit.os.path.exists')
    @patch('submission.submit.pd')
    @patch('submission.parseargs')
    def test_tta_submit(self, mock_parse_args, mock_read_csv, mock_exists):
        args=argparse.Namespace()
        args.folds=4
        args.work_dir=os.path.join(ARTIFACTS_ROOT,'tmp')
        mock_parse_args.parse_args.return_value=args
        mock_read_csv.read_csv.side_effect = read_scv_synth
        mock_exists.exists.return_value=True
        main()

        mock_read_csv.assert_called()





if __name__=="__main__":
    unittest.main()
