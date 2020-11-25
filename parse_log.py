import os
import pandas as pd
import glob
prefix='artifacts/val_quality_7_'
metrics_files=glob.glob(prefix+'*/metric.txt')

ms=[]
for file in metrics_files:
    df=pd.read_csv(file)
    df['name']=os.path.dirname(file)
    ms.append(df)

m=pd.concat(ms)
m.to_csv(prefix+'table.csv')
print(m.corr())
print(f'\n mean = \n {m.mean()}')
print(f' std = \n {m.std()}')

