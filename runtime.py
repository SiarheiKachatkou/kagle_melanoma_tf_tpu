

import os
import tensorflow as tf
import contextlib
from consts import *
from dataset_utils import *

print("Tensorflow version " + tf.__version__)

if use_amp:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    if not is_local:
        policy = mixed_precision.Policy('mixed_bfloat16')
    else:
        policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

def get_scope():
    if not is_debug:
        tpu_key='TPU_NAME'
        if tpu_key in os.environ:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ[tpu_key])
            tf.config.experimental_connect_to_cluster(resolver)
            # This is the TPU initialization code that has to be at the beginning.
            tf.tpu.experimental.initialize_tpu_system(resolver)
            print("All devices: ", tf.config.list_logical_devices('TPU'))
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
            scope=strategy.scope()
        else:
            print(f'{tpu_key} not found in {os.environ}')
            strategy = tf.distribute.MirroredStrategy()
            scope = strategy.scope()
    else:
        scope = contextlib.suppress()

    return scope