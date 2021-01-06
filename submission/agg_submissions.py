import pandas as pd
import os
import glob
from submission import aggregate_submissions

if __name__=="__main__":
    root='../subms'
    subms=glob.glob(root+'/*')
    print(f' found {len(subms)} submissions')
    subms=[pd.read_csv(s) for s in subms]
    for mode in ['AVGZ']:#,'MAX','MIN','STD']:
        agg_s=aggregate_submissions(subms, mode)
        agg_s.to_csv(root+f'/{mode}.csv',index=False)

