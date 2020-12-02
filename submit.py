import os
from sklearn import metrics
import pandas as pd
import numpy as np
import submission
import argparse
import glob



def calc_auc(subm):
    preds=subm['target'].values
    labels=subm['labels'].values
    if len(set(labels))==1:
        print('warning calc_auc with single label dataset, return 0')
        return 0
    return metrics.roc_auc_score(labels, preds)


def save_submission(df, name):
    df_submission = df[['image_name', 'target']]

    df_submission.to_csv(name, index=False)
    name_with_quotes='\"'+name+'\"'
    os.system(f'kaggle competitions submit -c siim-isic-melanoma-classification -f {name_with_quotes} -m {name_with_quotes}')

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir',type=str)
parser.add_argument('--folds',type=int, default=0)


if __name__=="__main__":

    args=parser.parse_args()

    if args.folds==0:
        nfolds = len(glob.glob(os.path.join(args.work_dir,'loss*.png')))
        print(f' --folds not specified, will use {nfolds}')
    else:
        nfolds=args.folds

    for le in ['', 'le']:
        for m_type in ['', 'tta_']:
            a = []

            for fold in range(nfolds):
                if len(le)>0:
                    name = f'val_le_{fold}_single_model_{m_type}submission.csv'
                else:
                    name = f'val_{fold}_single_model_{m_type}submission.csv'
                filename=os.path.join(args.work_dir, name)
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
                if le=='':
                    name = f'test_{fold}_single_model_{m_type}submission.csv'
                else:
                    name = f'test_{le}_{fold}_single_model_{m_type}submission.csv'
                filename=os.path.join(args.work_dir, name)
                if os.path.exists(filename):
                    sub = pd.read_csv(filename)
                    a.append(calc_auc(sub))
                    save_submission(sub, os.path.join(args.work_dir, 'kaggle_' + name))
                    subs.append(sub)
            if subs:
                avg_sub = submission.avg_submissions(subs)
                auc_avg_sub=calc_auc(avg_sub)
                save_submission(avg_sub, os.path.join(args.work_dir, 'kaggle_' + f'test_{le}_{m_type}.csv'))
            else:
                auc_avg_sub=None

            print(f'{le}_test_single_model_{m_type}metrics={a}')
            print(f'{le}_test_single_model_{m_type}avg_metric={np.mean(a)}')
            print(f'{le}_test_avg_model_{m_type}_metric={auc_avg_sub}')



