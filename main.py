

import os
import pickle
import tensorflow as tf
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__)
import tensorflow.keras.backend as K

from consts import *
from dataset_utils import *
import submission
from create_model import BinaryFocalLoss
from create_model import create_model

config=namedtuple('config',['lr_max','lr_start','lr_warm_up_epochs','lr_min','lr_exp_decay','nfolds','l2_penalty','model_fn_str','work_dir','ttas'])

CONFIG=config(lr_max=0.0002*8, lr_start=0.0002*8, lr_warm_up_epochs=0, lr_min=0.000005,lr_exp_decay=0.8, nfolds=4,l2_penalty=1e-6, work_dir='b6_focal_loss_768',
              model_fn_str="efficientnet.tfkeras.EfficientNetB0(weights='imagenet', include_top=False)", ttas=1)

#pretrained_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, 3], include_top=False)
#pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
# EfficientNet can be loaded through efficientnet.tfkeras library (https://github.com/qubvel/efficientnet)


if not is_debug:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ['TPU_NAME'])
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

if not os.path.exists(CONFIG.work_dir):
    os.mkdir(CONFIG.work_dir)
    
with open(os.path.join(CONFIG.work_dir,'config.pkl'),'wb') as file:
    pickle.dump(CONFIG,file)

    from lr import get_lrfn

lrfn=get_lrfn(CONFIG)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

train_filenames_folds, val_filenames_folds=get_train_val_filenames(DATASETS[IMAGE_HEIGHT],CONFIG.nfolds)
test_filenames=get_test_filenames(DATASETS[IMAGE_HEIGHT])
if is_debug:
    test_filenames=[test_filenames[0]]
    train_filenames_folds=[[f[0]] for f in train_filenames_folds]
    val_filenames_folds=[[f[0]] for f in val_filenames_folds]
    
test_dataset = get_test_dataset(test_filenames)
test_dataset_tta = get_test_dataset_tta(test_filenames)

for fold in range(CONFIG.nfolds):
    


    print(f'fold={fold}')
    
    if is_debug:
        model = create_model(CONFIG)
    else:
        with strategy.scope():
            model = create_model(CONFIG)

    model.summary()
    training_dataset = get_training_dataset(train_filenames_folds[fold])
    validation_dataset = get_validation_dataset(val_filenames_folds[fold])
    
    history = model.fit(return_2_values(training_dataset), validation_data=return_2_values(validation_dataset),
                        steps_per_epoch=TRAIN_STEPS, epochs=EPOCHS, callbacks=[lr_callback])

    final_accuracy = history.history["val_accuracy"][-5:]
    print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
    model.save(f'{CONFIG.work_dir}/model{fold}.h5')
    
    display_training_curves(history.history['accuracy'][1:], history.history['val_accuracy'][1:], 'accuracy', 211)
    display_training_curves(history.history['loss'][1:], history.history['val_loss'][1:], 'loss', 212)
    
    validation_dataset = get_validation_dataset(val_filenames_folds[fold])
    validation_dataset_tta = get_validation_dataset_tta(val_filenames_folds[fold])
    
    submission.calc_and_save_submissions(CONFIG,model,f'val_{fold}',validation_dataset, validation_dataset_tta, CONFIG.ttas)
    submission.calc_and_save_submissions(CONFIG,model,f'test_{fold}',test_dataset, test_dataset_tta,CONFIG.ttas)
    
    K.clear_session()