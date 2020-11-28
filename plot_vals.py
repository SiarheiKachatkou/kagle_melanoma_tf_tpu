import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
import consts


def parse_model_fn(s):
    for i in range(8):
        arch='B'+str(0)
        if arch in s:
            return arch
    return None

prefix='val_quality_9_'
val=pd.read_csv(prefix+'table.csv')
yaml_keys=['lr_max','model_fn_str','oversample_mult','dropout_rate','lr_exp_decay']
yaml_fns=[None,parse_model_fn,None,None,None]



def parse_name(name):
    with open(os.path.join(name,'config.yaml'),'rt') as file:
        config=yaml.load(file)
    vals={}
    for k,f in zip(yaml_keys,yaml_fns):
        v=config[k]
        if f is not None:
            v=f(v)
        vals[k]=v
    return vals

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