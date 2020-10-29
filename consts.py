import os
os.environ['TPU_NAME']='grpc://10.63.244.98:8470'


is_debug=True

EPOCHS = 1 if is_debug else 3

IMAGE_HEIGHT=768

IMAGE_SIZE=[IMAGE_HEIGHT, IMAGE_HEIGHT]

DATASETS = { # available image sizes
    128: 'gs://kaggle_melanoma_isic/isic2020-128-colornormed-tfrecord' +'/train*.tfrec',
    768: 'gs://kaggle_melanoma_isic/isic2020-768-colornormed-tfrecord/archive/train*.tfrec'
}

CLASSES = ['health','melanoma']

BATCH_SIZE = 4*8*4 if is_debug else 4*8*4

# TTEST IMAGES:  10982 , STEPS PER EPOCH:  343
# CPU
# B_S = 4 --- 47it [04:25,  5.65s/it]  345/(47*4)=1,8 sec/image
# b_S=32 -- 3it [02:31, 51.91s/it] 151/(3*32) = 1,5 sec/image = 4,7 hours

#TPU
#63it [04:13,  3.49s/it] 335/(128*63)=0,04 sec/image

TRAIN_STEPS = 10 if is_debug else 50000//BATCH_SIZE