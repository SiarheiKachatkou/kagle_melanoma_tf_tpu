import os
from sklearn import metrics
import pandas as pd
import numpy as np
from consts import CONFIG
import submission



def calc_auc(subm):
    preds=subm['target'].values
    labels=subm['labels'].values
    return metrics.roc_auc_score(labels, preds)


def save_submission(df, name):
    df_submission = df[['image_name', 'target']]

    df_submission.to_csv(name, index=False)
    os.system(f'kaggle competitions submit -c siim-isic-melanoma-classification -f {name} -m {name}')


if __name__=="__main__":
    src = CONFIG.work_dir

    for m_type in ['', 'tta_']:
        a = []

        for fold in range(CONFIG.nfolds):
            sub = pd.read_csv(os.path.join(src, f'val_{fold}_single_model_{m_type}submission.csv'))
            a.append(calc_auc(sub))
        print(f'single_model_{m_type}metrics={a}')
        print(f'single_model_{m_type}avg_metric={np.mean(a)}')

    for m_type in ['', 'tta_']:
        a = []
        subs = []
        for fold in range(CONFIG.nfolds):
            name = f'test_{fold}_single_model_{m_type}submission.csv'
            sub = pd.read_csv(os.path.join(src, name))
            save_submission(sub, os.path.join(src, 'kaggle_' + name))
            subs.append(sub)
        avg_sub = submission.avg_submissions(subs)
        save_submission(avg_sub, os.path.join(src, 'kaggle_' + f'test_{m_type}.csv'))


