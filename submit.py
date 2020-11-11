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
    return metrics.roc_auc_score(labels, preds)


def save_submission(df, name):
    df_submission = df[['image_name', 'target']]

    df_submission.to_csv(name, index=False)
    name_with_quotes='\"'+name+'\"'
    os.system(f'kaggle competitions submit -c siim-isic-melanoma-classification -f {name_with_quotes} -m {name_with_quotes}')

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir',type=str)


if __name__=="__main__":

    args=parser.parse_args()

    nfolds = len(glob.glob(os.path.join(args.work_dir,'loss*.png')))


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
            print(f'{le}_single_model_{m_type}metrics={a}')
            print(f'{le}_single_model_{m_type}avg_metric={np.mean(a)}')

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
                    save_submission(sub, os.path.join(args.work_dir, 'kaggle_' + name))
                    subs.append(sub)
            if subs:
                avg_sub = submission.avg_submissions(subs)
                save_submission(avg_sub, os.path.join(args.work_dir, 'kaggle_' + f'test_{le}_{m_type}.csv'))


