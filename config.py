import datetime
from collections import namedtuple
import os
import argparse
from consts import is_local, is_debug


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

parser.add_argument('--focal_loss_gamma',type=float,default=4)
parser.add_argument('--focal_loss_alpha',type=float,default=0.5)
parser.add_argument('--oversample_mult',type=int,default=1)

args=parser.parse_args()

if not is_local:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # "0" #
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


epochs_fine_tune = 0
epochs_full = 1 if is_debug else 20


BATCH_SIZE = 128 if is_debug else 64*3
BATCH_SIZE_INCREASE_FOR_INFERENCE = 16


TRAIN_STEPS = 1 if is_debug else None

config=namedtuple('config',['lr_max','lr_start','stepsize', 'lr_warm_up_epochs','lr_min','lr_exp_decay','lr_fn',
                            'nfolds','l2_penalty',
                            'model_fn_str','work_dir', 'gs_work_dir','ttas','use_metrics','dropout_rate',
                            'save_last_epochs',
                            'oversample_mult',
                            'focal_loss_gamma','focal_loss_alpha',
                            'hair_prob','microscope_prob',
                            'batch_size','batch_size_inference',
                            'image_height',
                            'epochs_full','epochs_fine_tune'
                            ])

model = args.backbone if not is_debug else 'B0'

penalty = 1e-6
dropout_rate=args.dropout_rate
focal_loss_alpha=args.focal_loss_alpha
focal_loss_gamma=args.focal_loss_gamma
hair_prob=args.hair_prob
microscope_prob=args.microscope_prob
lr_warm_up_epochs=args.lr_warm_up_epochs
image_height=args.image_height

work_dir_name = f'artifacts/tpu_{model}_focal_loss_{image_height}_epochs_{epochs_full}_drop_{dropout_rate}_lr_max{args.lr_max}_lr_dacay_{args.lr_exp_decay}_hair_prob_{hair_prob}_micro_prob_{microscope_prob}_wu_epochs_{lr_warm_up_epochs}' if not is_debug else 'debug'


CONFIG=config(lr_max=args.lr_max*1e-4, lr_start=5e-6, stepsize=3,
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
              batch_size=BATCH_SIZE, batch_size_inference=BATCH_SIZE * BATCH_SIZE_INCREASE_FOR_INFERENCE,
              image_height=image_height,
              epochs_full=epochs_full,epochs_fine_tune=epochs_fine_tune
              )

#pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
# EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)
