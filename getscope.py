import tensorflow as tf
import os

def get_scope():

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
        #print(f'{tpu_key} not found in {os.environ}')
        strategy = tf.distribute.MirroredStrategy()
        scope = strategy.scope()

    return scope