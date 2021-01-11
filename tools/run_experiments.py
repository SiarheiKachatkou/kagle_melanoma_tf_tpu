import os
import random
import multiprocessing as mp
from tqdm import tqdm
from itertools import product
from config.consts import is_local

hparams={'backbone':['B0','B1'], 'dropout_rate':[0.005,0.001], 'lr_max':[1, 0.1, 0.5], 'lr_exp_decay':[0.8,0.5],'hair_prob':[0,0.05, 0.1],'microscope_prob':[0,0.01],'lr_warm_up_epochs':[2,6,8],'image_height':[128,128], 'batch_size':[64], 'save_best_n':[1,2,4]}

keys=list(hparams.keys())
val_list=[hparams[k] for k in keys]
args=list(product(*val_list))
random.shuffle(args)
args=args[:3]

def get_gpu_available():
    worker_id=mp.current_process().name
    if worker_id=='MainProcess':
        return 0
    else:
        worker_id=worker_id.split('-')[1]
        gpu_id_for_process=int(worker_id)-1
        return gpu_id_for_process

def job(input_tuple):
    i, args_list=input_tuple
    cmd_string='python3 tools/main.py --gpus='+str(get_gpu_available())
    for k,v in zip(keys,args_list):
        cmd_string+=' --'+k+'='+str(v)
    cmd_string+=f' --stage={i} --work_dir=artifacts/{i}'

    os.system(cmd_string)

if is_local:
    num_procs=3
    pool=mp.Pool(num_procs)
    map_fn=pool.imap_unordered
else:
    map_fn=map

for _ in tqdm(map_fn(job, enumerate(args)), total=len(args)):
    pass
