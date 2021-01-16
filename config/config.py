import datetime
from collections import namedtuple
import os
import argparse
from config.consts import is_local, is_debug
from config.model_str_builder import build_model_str
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--backbone',type=str)
parser.add_argument('--dropout_rate',type=float)
parser.add_argument('--lr_max',type=float)
parser.add_argument('--lr_exp_decay',type=float)
parser.add_argument('--hair_prob',type=float)
parser.add_argument('--microscope_prob',type=float)
parser.add_argument('--lr_warm_up_epochs',type=int)
parser.add_argument('--gpus',type=str,default=None)
parser.add_argument('--image_height',type=int)
parser.add_argument('--work_dir',type=str)
parser.add_argument('--batch_size',type=int)
parser.add_argument('--stage',type=str)

parser.add_argument('--save_best_n',type=int,default=1)
parser.add_argument('--focal_loss_gamma',type=float,default=4)
parser.add_argument('--focal_loss_alpha',type=float,default=0.5)
parser.add_argument('--oversample_mult',type=int,default=1)
parser.add_argument('--cut_out_prob',type=float,default=0.2)
parser.add_argument('--cut_mix_prob',type=float,default=0.2)
parser.add_argument('--val_ttas',type=int,default=1)
parser.add_argument('--use_meta',type=int,default=0)
parser.add_argument('--epochs_full',type=int,default=12)

args=parser.parse_args()

if is_local:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"#"1,2"  # "0" #
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


epochs_fine_tune = 0
epochs_full = 1 if is_debug else epochs_fine_tune+args.epochs_full
epochs_total = epochs_full + 0

BATCH_SIZE = 36 if is_debug else args.batch_size
BATCH_SIZE_INCREASE_FOR_INFERENCE = 4


TRAIN_STEPS = 1 if is_debug else None

config=namedtuple('config',['lr_max','lr_start','lr_fine_tune','stepsize', 'lr_warm_up_epochs','lr_min','lr_exp_decay','lr_fn',
                            'nfolds','l2_penalty',
                            'model_fn_str','work_dir', 'gs_work_dir','ttas','use_metrics','dropout_rate',
                            'save_best_n',
                            'oversample_mult',
                            'focal_loss_gamma','focal_loss_alpha',
                            'hair_prob','microscope_prob','cut_out_prob','cut_mix_prob',
                            'batch_size','batch_size_inference',
                            'image_height',
                            'epochs_full','epochs_fine_tune', 'epochs_total', 'fine_tune_last',
                            'val_ttas',
                            'stage',
                            'use_meta'
                            ])

model = args.backbone if not is_debug else 'B0'

penalty = 0
dropout_rate=args.dropout_rate
focal_loss_alpha=args.focal_loss_alpha
focal_loss_gamma=args.focal_loss_gamma
hair_prob=args.hair_prob
microscope_prob=args.microscope_prob
lr_warm_up_epochs=args.lr_warm_up_epochs
image_height=args.image_height

ttas=2 if is_debug else 12


CONFIG=config(lr_max=args.lr_max*1e-4, lr_start=1e-6, stepsize=3,lr_fine_tune=3e-4,
              lr_warm_up_epochs=lr_warm_up_epochs,
              lr_min=1e-6, lr_exp_decay=args.lr_exp_decay, lr_fn='get_lrfn_fine_tune(CONFIG)',  #get_cycling_lrfn(CONFIG) #
              nfolds=4, l2_penalty=penalty, work_dir=args.work_dir,
              gs_work_dir=f'gs://kochetkov_kaggle_melanoma/{str(datetime.datetime.now())[:20]}_{args.work_dir}',
              model_fn_str=build_model_str(model),
              ttas=ttas,
              val_ttas=args.val_ttas,
              use_metrics=True, dropout_rate=dropout_rate,
              save_best_n=args.save_best_n,
              oversample_mult=args.oversample_mult,
              focal_loss_gamma=focal_loss_gamma, focal_loss_alpha=focal_loss_alpha,
              hair_prob=hair_prob, microscope_prob=microscope_prob,
              cut_out_prob=args.cut_out_prob,cut_mix_prob=args.cut_mix_prob,
              batch_size=BATCH_SIZE, batch_size_inference=BATCH_SIZE * BATCH_SIZE_INCREASE_FOR_INFERENCE,
              image_height=image_height,
              epochs_full=epochs_full,epochs_fine_tune=epochs_fine_tune, fine_tune_last=-1, epochs_total=epochs_total,
              stage=args.stage,
              use_meta=args.use_meta
              )

root=Path(os.path.split(__file__)[0])/'..'
metrics_path=root/'metrics'
if args.stage=='baseline':
    metrics_path/='metrics.txt'
else:
    metrics_path /= f'metrics_{args.stage}.txt'