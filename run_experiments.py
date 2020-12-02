import os
import random
import multiprocessing as mp
from tqdm import tqdm
from itertools import product

hparams={'backbone':['B0','B1'], 'dropout_rate':[0.005,0.001], 'lr_max':[8,20], 'lr_exp_decay':[0.8,0.5],'focal_loss_gamma':[2,4],
         'focal_loss_alpha':[0.5,0.8],'hair_prob':[0,0.05, 0.1,0.3],'microscope_prob':[0,0.01],'lr_warm_up_epochs':[2,6,8]}


keys=list(hparams.keys())
val_list=[hparams[k] for k in keys]
args=list(product(*val_list))
random.shuffle(args)

def get_gpu_available():
    worker_id=mp.current_process().name.split('-')[1]
    gpu_id_for_process=int(worker_id)-1
    return gpu_id_for_process

def job(args_list):
    cmd_string='python train_and_test.py --gpus='+str(get_gpu_available())
    for k,v in zip(keys,args_list):
        cmd_string+=' --'+k+'='+str(v)

    os.system(cmd_string)

num_procs=3
pool=mp.Pool(num_procs)
for _ in tqdm(pool.imap_unordered(job, args), total=len(args)):
    pass
