import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
import numpy as np
from plot_utils import parse_model_fn,parse_config,plot_pde


prefix='artifacts/val_quality_11_'
val=pd.read_csv(prefix+'table.csv')
yaml_parsers_fns={'model_fn_str':parse_model_fn}
yaml_ignore_keys={'gs_work_dir','work_dir','lr_fn'}
target_key='avg_test_auc'
features = [('hair_prob','microscope_prob'), 'hair_prob','microscope_prob','focal_loss_alpha','lr_warm_up_epochs']


vals_array=[]
yaml_keys=None
for i,name in enumerate(val.name.values):
    val_dict = parse_config(name,yaml_ignore_keys,yaml_parsers_fns)
    if yaml_keys is None:
        yaml_keys=val_dict.keys()
    vals_array.append([val_dict[k] for k in yaml_keys])
vals_array=np.array(vals_array)
for i,k in enumerate(yaml_keys):
    val[k]=vals_array[:,i]


rows=int(np.sqrt(len(features)))
cols=len(features)//rows+1
for i,f in enumerate(features):
    plt.subplot(rows,cols,i+1)
    plot_pde(val,f,target_key)
plt.show()
