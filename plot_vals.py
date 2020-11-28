import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
import numpy as np



def parse_model_fn(s):
    for i in range(8):
        arch='B'+str(i)
        if arch in s:
            return arch
    return None

prefix='artifacts/val_quality_9_'
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

vals_array=[]
for i,name in enumerate(val.name.values):
    val_dict = parse_name(name)
    vals_array.append([val_dict[k] for k in yaml_keys])
vals_array=np.array(vals_array)
for i,k in enumerate(yaml_keys):
    val[k]=vals_array[:,i]


dbg=1