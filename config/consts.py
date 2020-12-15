import os
import tensorflow as tf
import numpy as np
import random
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


use_tpu_2 = False
is_local = True
is_kaggle = False
is_debug = True
do_validate = True


use_amp=True

if (not is_local) and (not is_kaggle):
    tpu3 = "grpc://10.240.1.2:8470"
    tpu2 = 'grpc://10.240.1.10:8470'

    os.environ['TPU_NAME']=tpu2 if use_tpu_2 else tpu3

red = 4 if use_tpu_2 else 1
if is_local:
    red=4

root=Path(os.path.split(__file__)[0])/'..'

if is_local:
    DATASETS = {
                128: {'new': root/'data/128x128/train*.tfrec', 'old': ''},
                256: {'new': 'data/256_with_labels/train*.tfrec', 'old': ''},
                #128: {'new': 'data/128/train*.tfrec', 'old': ''},
                #384: {'new': 'data/isic2020-384-colornormed-tfrecord/train*.tfrec', 'old': ''},
                384: {'new': 'data/384_triple_2020_with_labels/train*.tfrec', 'old': ''},
                768: {'new': 'data/dataset_768/train*.tfrec', 'old': ''}
    }

else:
    DATASETS = {
                #128: {'new': 'gs://kaggle_melanoma_isic/isic2020-128-colornormed-tfrecord' +'/train*.tfrec', 'old':''},
                128: {'new': '/kaggle/input/melanoma-128x128/train*.tfrec','old':''},
                #128: {'new': 'data/128/train*.tfrec', 'old':''},
                #384: {'new': 'gs://kaggle_melanoma_isic/isic2020-384-colornormed-tfrecord/train*.tfrec', 'old':''},
                #384: {'new': 'data/isic2020-384-colornormed-tfrecord/train*.tfrec','old':''},
                #384: {'new':'gs://kds-76800f320871e548ef017f0a5a63cef5c72d1d47d6e020c81edfa286/train*.tfrec','old':''},
                384: {'new': 'gs://kds-68d604aee5addfebbabae3fcfe5376b86711f3cd42993bd8e0dc80a5/train*.tfrec','old':'gs://kds-4e8502fa6aa4c08b11f43ab8b42505960a29dc73fbcea54ba2bd1f9a/train*.tfrec'},
                #512: {'new': 'gs://kaggle_melanoma_isic/isic2020-512-colornormed-tfrecord/train*.tfrec', 'old':''},
                512: {'new': 'gs://kds-cb7c9aaea4b354a2b47b2c0f5feced50c9ddb98be03f7856e9e71642/train*.tfrec','old':''},
                768: {'new': 'gs://kaggle_melanoma_isic/isic2020-768-colornormed-tfrecord/archive/train*.tfrec',
                      'old': 'gs://kaggle_melanoma_isic/old-768-tfrecord/train*.tfrec'}
    }

CLASSES = ['health','melanoma']



seed=10000
op_seed=10
tf.random.set_seed(seed)
tf.compat.v1.random.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

path_hair_images = root/"data/melanoma_hairs" if is_local else ("gs://kds-c1f8d68ed78af3bc82472db8c32ec9b3fe1a0dcf09e62c04e90e81fe" if is_kaggle else "gs://kochetkov_kaggle_melanoma/melanoma_hairs")
path_microscope_images = root/"data/melanoma_microscope" if is_local else ("gs://kds-05490130d1f3d52b5eecc24712a9796337d072ebfbada94359adc410" if is_kaggle else "gs://kochetkov_kaggle_melanoma/melanoma_microscope" )

test_data_path=root/'data/test_data'

metrics_path=root/'metrics/metrics.txt'
