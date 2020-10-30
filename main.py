

import os
import pickle
import tensorflow as tf
import numpy as np
import contextlib
from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__)
import tensorflow.keras.backend as K

from consts import *
from dataset_utils import *
import submission
from create_model import BinaryFocalLoss
from create_model import create_model

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
            scope = contextlib.suppress()
    else:
        scope = contextlib.suppress()

    return scope

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
    test_filenames = [test_filenames[0]]
    train_filenames_folds=[[f[0]] for f in train_filenames_folds]
    val_filenames_folds=[[f[0]] for f in val_filenames_folds]

print(f'test filenames {test_filenames}')
test_dataset = get_test_dataset(test_filenames)
test_dataset_tta = get_test_dataset_tta(test_filenames)

for fold in range(CONFIG.nfolds):

    print(f'fold={fold}')

    scope = get_scope()

    with scope:
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

    submission.calc_and_save_submissions(CONFIG, model, f'test_{fold}', test_dataset, test_dataset_tta, CONFIG.ttas)
    submission.calc_and_save_submissions(CONFIG,model,f'val_{fold}',validation_dataset, validation_dataset_tta, CONFIG.ttas)

