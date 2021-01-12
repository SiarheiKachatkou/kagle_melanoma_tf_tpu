import re
from glob import glob
import os
import shutil

dst_root='.'
dst_metrics_root=os.path.join(dst_root,'metrics')
dst_artifacts_root=os.path.join(dst_root,'artifacts')

src_root='tpu3'

def get_metrics_paths(root):
    paths=glob(os.path.join(root, 'metrics','metrics_*.txt'))
    metrics=[p for p in paths if re.search('metrics_([0-9]*).txt',os.path.split(p)[1])]
    return metrics

def get_experiment_number_from_metric(metric_path):
    return int(re.findall('metrics_([0-9]*).txt', os.path.split(metric_path)[1])[0])

def get_max_experiment_number(metrics_paths):
    metrics_numbers = [ get_experiment_number_from_metric(p) for p in metrics_paths]
    return max(metrics_numbers)

src_metrics=get_metrics_paths(src_root)
dst_metrics=get_metrics_paths(dst_root)

metrics_number=get_max_experiment_number(dst_metrics)
metrics_number+=1
start_metric_number=metrics_number

for src_metric in src_metrics:
    src_artifact=os.path.join(src_root,'artifacts',str(get_experiment_number_from_metric(src_metric)))
    if os.path.exists(src_artifact):
        shutil.copyfile(src_metric,os.path.join(dst_metrics_root,f'metrics_{metrics_number}.txt'))
        dst_artifact=os.path.join(dst_artifacts_root,str(metrics_number))
        if os.path.exists(dst_artifact):
            shutil.rmtree(dst_artifact)
        shutil.copytree(src_artifact,dst_artifact)
        metrics_number+=1

print(f'{metrics_number-start_metric_number} new experiments copyed')
dbg=1


