import os
import glob
import pandas as pd
import numpy as np
from submission.submission_filenames import single_model_val_csv, single_model_test_csv
from submission.submission_utils import avg_submissions, calc_auc, save_submission
import submission.parseargs



def main():

    args=submission.parseargs.parse_args()

    if args.folds == 0:
        nfolds = len(glob.glob(os.path.join(args.work_dir, 'loss*.png')))
        print(f' --folds not specified, will use {nfolds}')
    else:
        nfolds = args.folds

    for le in ['', 'le']:
        for m_type in ['', 'tta_']:
            aucs = []

            for fold in range(nfolds):
                name=single_model_val_csv(le, fold, m_type)
                filename = os.path.join(args.work_dir, name)
                if os.path.exists(filename):
                    sub = pd.read_csv(filename)
                    aucs.append(calc_auc(sub))
            print(f'{le}_val_single_model_{m_type}metrics={aucs}')
            print(f'{le}_val_single_model_{m_type}avg_metric={np.mean(aucs)}')

    avg_test_subms=[]
    for le in ['', 'le']:
        for m_type in ['', 'tta_']:
            aucs = []
            subs = []
            for fold in range(nfolds):
                name=single_model_test_csv(le, fold, m_type)
                filename = os.path.join(args.work_dir, name)
                if os.path.exists(filename):
                    sub = pd.read_csv(filename)
                    aucs.append(calc_auc(sub))
                    save_submission(sub, os.path.join(args.work_dir, 'kaggle_' + name))
                    subs.append(sub)
            if subs:
                avg_sub = avg_submissions(subs)
                avg_test_subms.append(avg_sub)
                auc_avg_sub = calc_auc(avg_sub)
                save_submission(avg_sub, os.path.join(args.work_dir, 'kaggle_' + f'test_{le}_{m_type}.csv'))
            else:
                auc_avg_sub = None

            print(f'{le}_test_single_model_{m_type}metrics={aucs}')
            print(f'{le}_test_single_model_{m_type}avg_metric={np.mean(aucs)}')
            print(f'{le}_test_avg_model_{m_type}_metric={auc_avg_sub}')

    return avg_test_subms


if __name__=="__main__":
    main()


