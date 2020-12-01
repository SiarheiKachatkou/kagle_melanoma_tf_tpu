import multiprocessing as mp
from itertools import permutations

hparams={'backbone':['B0','B1'], 'dropout_rates':[0.01,0], 'lr_max':[10,30], 'lr_exp_decay':[0.8,0.5],'focal_loss_gamma':[2,4],
         'focal_loss_alpha':[0.5,0.8],'hair_prob':[0,0.1,0.3],'microscope_prob':[0,0.01,0.02],'lr_warm_up_epochs':[0,5,7]}


gpus=['0','1','2']



python train_and_test.py --backbone=$backbone --dropout-rate=$dropout_rate --lr_max=$lr_max --lr_exp_decay=$lr_exp_decay --focal_loss_gamma=$focal_loss_gamma
--focal_loss_alpha=$focal_loss_alpha --hair-prob=$hair_prob --microscope-prob=$micro_prob --lr-warm-up-epochs=$lr_warm_up_epochs