import datetime
from collections import namedtuple
import os
from config import parseargs
from config.consts import is_local, is_debug



epochs_fine_tune = 0
epochs_full = 1 if is_debug else 20


BATCH_SIZE = 128 if is_debug else 64*4
BATCH_SIZE_INCREASE_FOR_INFERENCE = 16


TRAIN_STEPS = 1 if is_debug else None

config_type=namedtuple('config_type',['lr_max','lr_start','stepsize', 'lr_warm_up_epochs','lr_min','lr_exp_decay','lr_fn',
                            'nfolds','l2_penalty',
                            'model_fn_str','work_dir', 'gs_work_dir','ttas','use_metrics','dropout_rate',
                            'save_last_epochs',
                            'oversample_mult',
                            'focal_loss_gamma','focal_loss_alpha',
                            'hair_prob','microscope_prob',
                            'batch_size','batch_size_inference','train_steps',
                            'image_height',
                            'epochs_full','epochs_fine_tune'
                            ])


def get_config():

    args = parseargs.parse_args()

    if not is_local:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if args.gpus is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # "0" #
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    model = args.backbone if not is_debug else 'B0'

    penalty = 1e-6
    dropout_rate=args.dropout_rate
    focal_loss_alpha=args.focal_loss_alpha
    focal_loss_gamma=args.focal_loss_gamma
    hair_prob=args.hair_prob
    microscope_prob=args.microscope_prob
    lr_warm_up_epochs=args.lr_warm_up_epochs
    image_height=args.image_height

    work_dir_name = f'artifacts/val_quality_13_{model}_focal_loss_{image_height}_epochs_{epochs_full}_drop_{dropout_rate}_lr_max{args.lr_max}_lr_dacay_{args.lr_exp_decay}_hair_prob_{hair_prob}_micro_prob_{microscope_prob}_wu_epochs_{lr_warm_up_epochs}' \
        if not is_debug else 'artifacts/debug'


    config=config_type(lr_max=args.lr_max*1e-4, lr_start=5e-6, stepsize=3,
                  lr_warm_up_epochs=lr_warm_up_epochs,
                  lr_min=1e-6, lr_exp_decay=args.lr_exp_decay, lr_fn='get_lrfn(CONFIG)',  #get_cycling_lrfn(CONFIG) #
                  nfolds=4, l2_penalty=penalty, work_dir=work_dir_name,
                  gs_work_dir=f'gs://kochetkov_kaggle_melanoma/{str(datetime.datetime.now())[:20]}_{work_dir_name}',
                  model_fn_str=f"efficientnet.tfkeras.EfficientNet{model}(weights='imagenet', include_top=False)",
                  ttas=12,
                  use_metrics=True, dropout_rate=dropout_rate,
                  save_last_epochs=0,
                  oversample_mult=args.oversample_mult,
                  focal_loss_gamma=focal_loss_gamma, focal_loss_alpha=focal_loss_alpha,
                  hair_prob=hair_prob, microscope_prob=microscope_prob,
                  batch_size=BATCH_SIZE, batch_size_inference=BATCH_SIZE * BATCH_SIZE_INCREASE_FOR_INFERENCE,train_steps=TRAIN_STEPS,
                  image_height=image_height,
                  epochs_full=epochs_full,epochs_fine_tune=epochs_fine_tune
                  )

    return config