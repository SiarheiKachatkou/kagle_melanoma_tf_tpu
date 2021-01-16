import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from debug_tools.pde.plot_utils import parse_model_fn,plot_pde
from debug_tools.pde.parse_log import parse_logs


yaml_parsers_fns={'model_fn_str':parse_model_fn}
yaml_ignore_keys={'gs_work_dir','work_dir','lr_fn'}
target_key='test_avg_tta_auc'


features = ['model_fn_str',
            'use_meta','cut_mix_prob','cut_out_prob']

experiments_root='/mnt/850G/GIT/kaggle_melanoma_experiments/experiments_use_meta_e997306/'

df=parse_logs(experiments_root+'metrics/metrics_*.txt',experiments_root+'artifacts',yaml_ignore_keys,yaml_parsers_fns)

#df=df[df['model_fn_str']==2]

rows=int(np.sqrt(len(features)))
cols=len(features)//rows+1
for i,f in enumerate(features):
    plt.subplot(rows,cols,i+1)
    plot_pde(df,f,target_key)
plt.show()
