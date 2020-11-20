import os
import pandas as pd
import glob
metrics_files=glob.glob('val_quality_*/metric.txt')

ms=[]
for file in metrics_files:
    df=pd.read_csv(file)
    df['name']=os.path.dirname(file)
    ms.append(df)

m=pd.concat(ms)
m.to_csv('table.csv')
print(m.corr())

