import os
import pandas as pd
import glob
import numpy as np
import yaml
import re


def parse_metrics(path_pattern):

    metrics_files=glob.glob(path_pattern)

    ms=[]
    for file in metrics_files:
        with open(file, 'rt') as f:
            y = yaml.load(f)
        df=pd.DataFrame(columns=y.keys(), data=[y.values()])
        df['name']=file
        ms.append(df)

    m=pd.concat(ms)
    print(m.corr())
    print(f'\n mean = \n {m.mean()}')
    print(f' std = \n {m.std()}')
    return m


def parse_config(name, yaml_ignore_keys,  yaml_parsers_fns):
    with open(os.path.join(name,'config.yaml'),'rt') as file:
        config=yaml.load(file)
    vals={}
    for k,f in config.items():
        if k in yaml_ignore_keys:
            continue
        f=yaml_parsers_fns[k] if k in yaml_parsers_fns else None
        v=config[k]
        if f is not None:
            v=f(v)
        else:
            try:
                v=float(v)
            except:
                pass
        vals[k]=v
    return vals


def add_values_from_configs(val,yaml_ignore_keys, yaml_parsers_fns):
    vals_array = []
    yaml_keys = None
    for i, name in enumerate(val.name.values):
        val_dict = parse_config(name, yaml_ignore_keys, yaml_parsers_fns)
        if yaml_keys is None:
            yaml_keys = val_dict.keys()
        vals_array.append([val_dict[k] for k in yaml_keys])
    vals_array = np.array(vals_array)
    for i, k in enumerate(yaml_keys):
        val[k] = vals_array[:, i]
    return val

def parse_logs(metric_path_pattern, artifacts_root, yaml_ignore_keys,yaml_parsers_fns):
    df=parse_metrics(metric_path_pattern)
    def _metric_name_to_artifact(metric_path):
        r=re.findall('.*_([0-9]*).txt',os.path.split(metric_path)[1])
        return os.path.join(artifacts_root,r[0])

    df['name'] = df['name'].apply(_metric_name_to_artifact)

    df=add_values_from_configs(df,yaml_ignore_keys,yaml_parsers_fns)
    return df
