import datetime
from collections import namedtuple
import os
import argparse
from config.consts import is_local, is_debug


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

parser.add_argument('--focal_loss_gamma',type=float,default=4)
parser.add_argument('--focal_loss_alpha',type=float,default=0.5)
parser.add_argument('--oversample_mult',type=int,default=1)

args=parser.parse_args()

if is_local:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"#"1,2"  # "0" #
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


epochs_fine_tune = 12
epochs_full = 1 if is_debug else epochs_fine_tune+24


BATCH_SIZE = 36 if is_debug else 512
BATCH_SIZE_INCREASE_FOR_INFERENCE = 4


TRAIN_STEPS = 1 if is_debug else None

config=namedtuple('config',['lr_max','lr_start','lr_fine_tune','stepsize', 'lr_warm_up_epochs','lr_min','lr_exp_decay','lr_fn',
                            'nfolds','l2_penalty',
                            'model_fn_str','work_dir', 'gs_work_dir','ttas','use_metrics','dropout_rate',
                            'save_last_epochs',
                            'oversample_mult',
                            'focal_loss_gamma','focal_loss_alpha',
                            'hair_prob','microscope_prob','cut_out_prob','cut_mix_prob',
                            'batch_size','batch_size_inference',
                            'image_height',
                            'epochs_full','epochs_fine_tune', 'fine_tune_last'
                            ])

model = args.backbone if not is_debug else 'B0'

penalty = 1e-4
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
              model_fn_str=f"efficientnet.tfkeras.EfficientNet{model}(weights='imagenet', include_top=False)",
              ttas=ttas,
              use_metrics=True, dropout_rate=dropout_rate,
              save_last_epochs=0,
              oversample_mult=args.oversample_mult,
              focal_loss_gamma=focal_loss_gamma, focal_loss_alpha=focal_loss_alpha,
              hair_prob=hair_prob, microscope_prob=microscope_prob,cut_out_prob=0.1,cut_mix_prob=0.05,
              batch_size=BATCH_SIZE, batch_size_inference=BATCH_SIZE * BATCH_SIZE_INCREASE_FOR_INFERENCE,
              image_height=image_height,
              epochs_full=epochs_full,epochs_fine_tune=epochs_fine_tune, fine_tune_last=-1
              )

#pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
# EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)
