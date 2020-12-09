import os

import pandas as pd
import numpy as np
from submission.submission_filenames import single_model_val_csv, single_model_test_csv
from submission.submission import avg_submissions, calc_auc, save_submission

import argparse
import glob




parser = argparse.ArgumentParser()
parser.add_argument('--work_dir',type=str)
parser.add_argument('--folds',type=int, default=0)


def main():

    args = parser.parse_args()

    if args.folds == 0:
        nfolds = len(glob.glob(os.path.join(args.work_dir, 'loss*.png')))
        print(f' --folds not specified, will use {nfolds}')
    else:
        nfolds = args.folds

    for le in ['', 'le']:
        for m_type in ['', 'tta_']:
            a = []

            for fold in range(nfolds):
                name=single_model_val_csv(le, fold, m_type)
                filename = os.path.join(args.work_dir, name)
                if os.path.exists(filename):
                    sub = pd.read_csv(filename)
                    a.append(calc_auc(sub))
            print(f'{le}_val_single_model_{m_type}metrics={a}')
            print(f'{le}_val_single_model_{m_type}avg_metric={np.mean(a)}')

    for le in ['', 'le']:
        for m_type in ['', 'tta_']:
            a = []
            subs = []
            for fold in range(nfolds):
                name=single_model_test_csv(le, fold, m_type)
                filename = os.path.join(args.work_dir, name)
                if os.path.exists(filename):
                    sub = pd.read_csv(filename)
                    a.append(calc_auc(sub))
                    save_submission(sub, os.path.join(args.work_dir, 'kaggle_' + name))
                    subs.append(sub)
            if subs:
                avg_sub = avg_submissions(subs)
                auc_avg_sub = calc_auc(avg_sub)
                save_submission(avg_sub, os.path.join(args.work_dir, 'kaggle_' + f'test_{le}_{m_type}.csv'))
            else:
                auc_avg_sub = None

            print(f'{le}_test_single_model_{m_type}metrics={a}')
            print(f'{le}_test_single_model_{m_type}avg_metric={np.mean(a)}')
            print(f'{le}_test_avg_model_{m_type}_metric={auc_avg_sub}')


if __name__=="__main__":
    main()


