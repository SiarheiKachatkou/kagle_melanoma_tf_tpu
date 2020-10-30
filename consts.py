import datetime
from collections import namedtuple
#import os
#os.environ['TPU_NAME']='grpc://10.63.244.98:8470'


is_debug = False

EPOCHS = 1 if is_debug else 1

IMAGE_HEIGHT = 128

IMAGE_SIZE=[IMAGE_HEIGHT, IMAGE_HEIGHT]

DATASETS = { # available image sizes
    128: 'gs://kaggle_melanoma_isic/isic2020-128-colornormed-tfrecord' +'/train*.tfrec',
    768: 'gs://kaggle_melanoma_isic/isic2020-768-colornormed-tfrecord/archive/train*.tfrec'
}

CLASSES = ['health','melanoma']

BATCH_SIZE = 2 if is_debug else 4*8*2

# TTEST IMAGES:  10982 , STEPS PER EPOCH:  343
# CPU
# B_S = 4 --- 47it [04:25,  5.65s/it]  345/(47*4)=1,8 sec/image
# b_S=32 -- 3it [02:31, 51.91s/it] 151/(3*32) = 1,5 sec/image = 4,7 hours

#TPU
#63it [04:13,  3.49s/it] 335/(128*63)=0,04 sec/image

TRAIN_STEPS = 1 if is_debug else 10 #50000//BATCH_SIZE



config=namedtuple('config',['lr_max','lr_start','lr_warm_up_epochs','lr_min','lr_exp_decay','nfolds','l2_penalty','model_fn_str','work_dir', 'gs_work_dir','ttas'])

work_dir_name='b6_focal_loss_768'

CONFIG=config(lr_max=0.0002*8, lr_start=0.0002*8, lr_warm_up_epochs=0, lr_min=0.000005,lr_exp_decay=0.8, nfolds=4,
              l2_penalty=1e-6, work_dir=work_dir_name,
              gs_work_dir=f'gs://kochetkov_kaggle_melanoma/{work_dir_name}+{str(datetime.datetime.now())}',
              model_fn_str="efficientnet.tfkeras.EfficientNetB0(weights='imagenet', include_top=False)", ttas=1,
              )

#pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
# EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)

