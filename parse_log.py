import os
import pandas as pd
import glob
prefix='val_quality_2_'
metrics_files=glob.glob(prefix+'*/metric.txt')

ms=[]
for file in metrics_files:
    df=pd.read_csv(file)
    df['name']=os.path.dirname(file)
    ms.append(df)

m=pd.concat(ms)
m.to_csv(prefix+'table.csv')
print(m.corr())

