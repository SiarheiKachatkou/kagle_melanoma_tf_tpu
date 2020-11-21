import datetime
from collections import namedtuple
import os
import tensorflow as tf
import numpy as np
import random
import sklearn
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--backbone',type=str)
parser.add_argument('--cut-mix-prob',type=float)
parser.add_argument('--dropout-rate',type=float)
args=parser.parse_args()


use_tpu_2 = False
is_local = True
is_kaggle = False
is_debug = False
use_amp = True

if (not is_local) and (not is_kaggle):
    tpu3 = "grpc://10.240.1.2:8470"
    tpu2 = 'grpc://10.240.1.10:8470'

    os.environ['TPU_NAME']=tpu2 if use_tpu_2 else tpu3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


EPOCHS_FINE_TUNE = 2
EPOCHS_FULL = 1 if is_debug else 8

IMAGE_HEIGHT = 128

IMAGE_SIZE=[IMAGE_HEIGHT, IMAGE_HEIGHT]

if is_local:
    DATASETS = {
                128: {'new': 'data/128_with_labels/train*.tfrec', 'old': ''},
                256: {'new': 'data/256_with_labels/train*.tfrec', 'old': ''},
                #128: {'new': 'data/128/train*.tfrec', 'old': ''},
                #384: {'new': 'data/isic2020-384-colornormed-tfrecord/train*.tfrec', 'old': ''},
                384: {'new': 'data/384_triple_2020/train*.tfrec', 'old': ''},
                768: {'new': 'data/dataset_768/train*.tfrec', 'old': ''}
    }

else:
    DATASETS = {
                128: {'new': 'gs://kaggle_melanoma_isic/isic2020-128-colornormed-tfrecord' +'/train*.tfrec', 'old':''},
                #128: {'new': 'data/128/train*.tfrec', 'old':''},
                #384: {'new': 'gs://kaggle_melanoma_isic/isic2020-384-colornormed-tfrecord/train*.tfrec', 'old':''},
                #384: {'new': 'data/isic2020-384-colornormed-tfrecord/train*.tfrec','old':''},
                384: {'new':'gs://kds-76800f320871e548ef017f0a5a63cef5c72d1d47d6e020c81edfa286/train*.tfrec','old':''},
                512: {'new': 'gs://kaggle_melanoma_isic/isic2020-512-colornormed-tfrecord/train*.tfrec', 'old':''},
                768: {'new': 'gs://kaggle_melanoma_isic/isic2020-768-colornormed-tfrecord/archive/train*.tfrec',
                      'old': 'gs://kaggle_melanoma_isic/old-768-tfrecord/train*.tfrec'}
    }

CLASSES = ['health','melanoma']

red = 4 if use_tpu_2 else 1

if is_local:
    red=4

BATCH_SIZE = 128 if is_debug else 512

TRAIN_STEPS = 1 if is_debug else None

config=namedtuple('config',['lr_max','lr_start','stepsize', 'lr_warm_up_epochs','lr_min','lr_exp_decay','lr_fn',
                            'nfolds','l2_penalty',
                            'model_fn_str','work_dir', 'gs_work_dir','ttas','use_metrics','dropout_rate',
                            'save_last_epochs',
                            'cut_mix_prob'])

model = args.backbone if not is_debug else 'B0'

penalty = 0
cut_mix_prob=args.cut_mix_prob
dropout_rate=args.dropout_rate

work_dir_name = f'val_quality_4_{model}_bce_loss_{IMAGE_HEIGHT}_epochs_{EPOCHS_FULL}_cut_mix_{cut_mix_prob}_drop_{dropout_rate}' if not is_debug else 'debug'


CONFIG=config(lr_max=3e-4, lr_start=5e-6, stepsize=3, lr_warm_up_epochs=5, lr_min=1e-6,lr_exp_decay=0.8,lr_fn='get_lrfn(CONFIG)',#get_cycling_lrfn(CONFIG) #
              nfolds=4, l2_penalty=penalty, work_dir=work_dir_name,
              gs_work_dir=f'gs://kochetkov_kaggle_melanoma/{str(datetime.datetime.now())[:20]}_{work_dir_name}',
              model_fn_str=f"efficientnet.tfkeras.EfficientNet{model}(weights='imagenet', include_top=False)", ttas=6,
              use_metrics=True, dropout_rate=dropout_rate,
              save_last_epochs=0,
              cut_mix_prob=cut_mix_prob
              )

#pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
# EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)

seed=10000
op_seed=10
tf.random.set_seed(seed)
tf.compat.v1.random.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)
