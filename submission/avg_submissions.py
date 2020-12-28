import pandas as pd
import os
import glob
from submission import avg_submissions

if __name__=="__main__":
    root='../../subms'
    subms=glob.glob(root+'/*')
    subms=[pd.read_csv(s) for s in subms]
    avg_s=avg_submissions(subms)
    avg_s.to_csv(root+'/avg.csv',index=False)

