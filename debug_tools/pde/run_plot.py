import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from debug_tools.pde.plot_utils import parse_model_fn,plot_pde
from debug_tools.pde.parse_log import parse_logs


yaml_parsers_fns={'model_fn_str':parse_model_fn}
yaml_ignore_keys={'gs_work_dir','work_dir','lr_fn'}
target_key='test_avg_tta_auc'


features = [('cut_out_prob','cut_mix_prob'),('cut_out_prob','model_fn_str'),('cut_mix_prob','model_fn_str'),'model_fn_str','hair_prob','microscope_prob',
            'dropout_rate','cut_out_prob','cut_mix_prob','image_height']

df=parse_logs('metrics/metrics_*.txt','artifacts',yaml_ignore_keys,yaml_parsers_fns)


rows=int(np.sqrt(len(features)))
cols=len(features)//rows+1
for i,f in enumerate(features):
    plt.subplot(rows,cols,i+1)
    plot_pde(df,f,target_key)
plt.show()
