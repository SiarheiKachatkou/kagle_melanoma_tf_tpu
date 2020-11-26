import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestRegressor

prefix='artifacts/val_quality_5_'
val=pd.read_csv(prefix+'table.csv')

def get_hparams(name):#TODO read data from saved namedtuple config
    ret=re.findall(prefix+'([A-Z0-9]*)_(.*)_cut_mix_([0-9\.]*)_drop_([0-9\.]*)',name)
    arch=ret[0][0]
    cut_mix=float(ret[0][2])
    drop=float(ret[0][3])
    return arch,cut_mix,drop

archs=[None]*len(val)
cut_mixs=[None]*len(val)
drops=[None]*len(val)
for i,name in enumerate(val.name.values):
    archs[i], cut_mixs[i], drops[i] = get_hparams(name)

val['cut_mix']=cut_mixs
val['drop']=drops
val['arch']=archs

target_key='avg_test_auc'
features_keys=['cut_mix', 'drop', 'arch']

for v,n in zip(['B0','B1','B2','B3'],[0,1,2,3]):
    val['arch'].iloc[val['arch']==v]=n

Y=val[target_key].values

clf = RandomForestRegressor(n_estimators=100, random_state=0).fit(val[features_keys], Y)
features = ['cut_mix', 'drop','arch', ('drop', 'arch')]

plot_partial_dependence(clf, val[features_keys], features)
plt.show()

val[(val['arch']=='B1') & (val['drop']==0)][['val_auc','with_augm_auc','avg_test_auc','test_auc','cut_mix']]

dbg=1