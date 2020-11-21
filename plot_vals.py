import matplotlib.pyplot as plt
import pandas as pd
import re

val=pd.read_csv('val_quality_2_table.csv')

arch='B0'
drop='0'

def parse_name(name):
    ret=re.findall('val_quality_2_([A-Z0-9]*)_(.*)_cut_mix_([0-9\.]*)_drop_([0-9\.]*)',name)
    arch=ret[0][0]
    cut_mix=float(ret[0][2])
    drop=float(ret[0][3])
    return arch,cut_mix,drop

archs=[None]*len(val)
cut_mixs=[None]*len(val)
drops=[None]*len(val)
for i,name in enumerate(val.name.values):
    archs[i], cut_mixs[i], drops[i] = parse_name(name)

val['cut_mix']=cut_mixs
val['drop']=drops
val['arch']=archs

val[(val['arch']=='B1') & (val['drop']==0)][['val_auc','with_augm_auc','avg_test_auc','test_auc','cut_mix']]

dbg=1