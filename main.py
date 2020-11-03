

import os
import pickle
import tensorflow as tf
import numpy as np
import subprocess
import contextlib
from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__)
from lr import get_lrfn
from display_utils import display_training_curves
from consts import *
from dataset_utils import *
import submission
import shutil
from create_model import BinaryFocalLoss
from create_model import create_model, set_backbone_trainable


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
            scope = contextlib.suppress()
    else:
        scope = contextlib.suppress()

    return scope

if not os.path.exists(CONFIG.work_dir):
    os.mkdir(CONFIG.work_dir)
    
shutil.copyfile('consts.py',os.path.join(CONFIG.work_dir,'consts.py'))

lrfn=get_lrfn(CONFIG)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
rng = [i for i in range(EPOCHS_FULL)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
plt.title("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
plt.savefig(os.path.join(CONFIG.work_dir,'lr_schedule.png'))

train_filenames_folds, val_filenames_folds=get_train_val_filenames(DATASETS[IMAGE_HEIGHT]['new'],CONFIG.nfolds)
test_filenames=get_test_filenames(DATASETS[IMAGE_HEIGHT]['new'])
if is_debug:
    test_filenames = [test_filenames[0]]
    train_filenames_folds=[[f[0]] for f in train_filenames_folds]
    val_filenames_folds=[[f[0]] for f in val_filenames_folds]


for fold in range(CONFIG.nfolds):

    print(f'fold={fold}')
    scope = get_scope()
    with scope:
        model = create_model(CONFIG, backbone_trainable=False)

    model.summary()
    training_dataset = get_training_dataset(train_filenames_folds[fold], DATASETS[IMAGE_HEIGHT]['old'])
    validation_dataset = get_validation_dataset(val_filenames_folds[fold])
    
    history_fine_tune = model.fit(return_2_values(training_dataset),
                                  validation_data=return_2_values(validation_dataset), steps_per_epoch=TRAIN_STEPS,
                                  epochs=EPOCHS_FINE_TUNE, callbacks=[lr_callback])

    model = set_backbone_trainable(model, True, CONFIG)

    history = model.fit(return_2_values(training_dataset), validation_data=return_2_values(validation_dataset),
                        steps_per_epoch=TRAIN_STEPS, initial_epoch=EPOCHS_FINE_TUNE, epochs=EPOCHS_FULL, callbacks=[lr_callback])

    final_accuracy = history.history["val_accuracy"][-5:]
    print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
    model.save(f'{CONFIG.work_dir}/model{fold}.h5')
    print(history_fine_tune.history)
    print(history.history)

    if CONFIG.use_metrics:
        display_training_curves(history.history['auc'][1:], history.history['val_auc'][1:], 'auc', 211)
        plt.savefig(os.path.join(CONFIG.work_dir, f'auc_{fold}.png'))
    display_training_curves(history.history['loss'][1:], history.history['val_loss'][1:], 'loss', 212)
    plt.savefig(os.path.join(CONFIG.work_dir, f'loss{fold}.png'))

    test_dataset = get_test_dataset(test_filenames)
    test_dataset_tta = get_test_dataset_tta(test_filenames)

    validation_dataset = get_validation_dataset(val_filenames_folds[fold])
    validation_dataset_tta = get_validation_dataset_tta(val_filenames_folds[fold])

    submission.calc_and_save_submissions(CONFIG, model, f'val_{fold}', validation_dataset, validation_dataset_tta,
                                         CONFIG.ttas)
    submission.calc_and_save_submissions(CONFIG, model, f'test_{fold}', test_dataset, test_dataset_tta, CONFIG.ttas)

    subprocess.check_call(['gsutil', '-m', 'cp', '-r', CONFIG.work_dir,CONFIG.gs_work_dir])
