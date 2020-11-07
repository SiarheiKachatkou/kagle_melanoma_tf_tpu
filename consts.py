import datetime
from collections import namedtuple
import os


use_tpu_2 = False

tpu3 = 'grpc://10.240.1.2:8470'
tpu2 = "grpc://10.240.1.10:8470"

os.environ['TPU_NAME']=tpu2 if use_tpu_2 else tpu3

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

is_debug = True

EPOCHS_FINE_TUNE = 1
EPOCHS_FULL = 5 if is_debug else 10

IMAGE_HEIGHT = 128

IMAGE_SIZE=[IMAGE_HEIGHT, IMAGE_HEIGHT]

DATASETS = {
            128: {'new': 'gs://kaggle_melanoma_isic/isic2020-128-colornormed-tfrecord' +'/train*.tfrec', 'old':''},
            512: {'new': 'gs://kaggle_melanoma_isic/isic2020-512-colornormed-tfrecord/train*.tfrec', 'old':''},
            768: {'new': 'gs://kaggle_melanoma_isic/isic2020-768-colornormed-tfrecord/archive/train*.tfrec',
                  'old': 'gs://kaggle_melanoma_isic/old-768-tfrecord/train*.tfrec'}
}

#'dataset_768/train*.tfrec',

CLASSES = ['health','melanoma']

red = 4 if use_tpu_2 else 1

BATCH_SIZE = 1 if is_debug else 8*4//red

TRAIN_STEPS = 1 if is_debug else 50000//BATCH_SIZE

config=namedtuple('config',['lr_max','lr_start','lr_warm_up_epochs','lr_min','lr_exp_decay','nfolds','l2_penalty',
                            'model_fn_str','work_dir', 'gs_work_dir','ttas','use_metrics','dropout_rate',
                            'save_last_epochs'])

work_dir_name = f'b6_focal_loss_{IMAGE_HEIGHT}_old_datasets_penalty_1e-6'


CONFIG=config(lr_max=0.0002*8/red, lr_start=0.0002*8/red, lr_warm_up_epochs=0, lr_min=0.000005/red,lr_exp_decay=0.8,
              nfolds=4, l2_penalty=1e-6, work_dir=work_dir_name,
              gs_work_dir=f'gs://kochetkov_kaggle_melanoma/{work_dir_name}_{str(datetime.datetime.now())}',
              model_fn_str="efficientnet.tfkeras.EfficientNetB0(weights='imagenet', include_top=False)", ttas=1,
              use_metrics=True, dropout_rate=0.5,
              save_last_epochs=3
              )

#pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
# EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)

