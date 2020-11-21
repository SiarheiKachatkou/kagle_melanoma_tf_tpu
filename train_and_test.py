

import os
import glob
import gc
import tensorflow as tf
import subprocess
from matplotlib import pyplot as plt
from lr import get_lrfn, get_cycling_lrfn
from display_utils import display_training_curves, plot_lr
from consts import *
from dataset_utils import *
import submission
import shutil
import pandas as pd
from create_model import BinaryFocalLoss
from SaveLastCallback import SaveLastCallback
from create_model import create_model, set_backbone_trainable
from runtime import get_scope
from history import join_history
from submit import calc_auc

if not os.path.exists(CONFIG.work_dir):
    os.mkdir(CONFIG.work_dir)
    
shutil.copyfile('consts.py',os.path.join(CONFIG.work_dir,'consts.py'))

lrfn = eval(CONFIG.lr_fn)
plot_lr(lrfn,EPOCHS_FULL,CONFIG.work_dir)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

train_filenames_folds, val_filenames_folds=get_train_val_filenames(DATASETS[IMAGE_HEIGHT]['new'],CONFIG.nfolds)
test_filenames=get_test_filenames(DATASETS[IMAGE_HEIGHT]['new'])
if is_debug:
    test_filenames = [test_filenames[0]]
    train_filenames_folds=[[f[0]] for f in train_filenames_folds]
    val_filenames_folds=[[f[0]] for f in val_filenames_folds]


for fold in range(CONFIG.nfolds):
    print(f'fold={fold}')
    model_file_path=f'{CONFIG.work_dir}/model{fold}.h5'
    # SAVE BEST MODEL EACH FOLD
    save_callback_best = tf.keras.callbacks.ModelCheckpoint(
        model_file_path, monitor='val_loss', verbose=0, save_best_only=True,
        mode='min', save_freq='epoch')
    save_callback_last=SaveLastCallback(CONFIG.work_dir, fold, EPOCHS_FULL, CONFIG.save_last_epochs)

    save_callback=SaveLastCallback(CONFIG.work_dir,fold=fold, epochs=EPOCHS_FULL,
                                   save_last_epochs=CONFIG.save_last_epochs)

    callbacks=[lr_callback,save_callback_best,save_callback_last]

    scope = get_scope()
    with scope:
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')] if CONFIG.use_metrics else None
        model = create_model(CONFIG, metrics, backbone_trainable=False)

        model.summary()
        training_dataset = get_training_dataset(train_filenames_folds[fold], DATASETS[IMAGE_HEIGHT]['old'], CONFIG)
        if TRAIN_STEPS is None:
            TRAIN_STEPS=count_data_items(train_filenames_folds[fold])//BATCH_SIZE
        print(f'TRAIN_STEPS={TRAIN_STEPS}')
        validation_dataset = get_validation_dataset(val_filenames_folds[fold])

        history_fine_tune = model.fit(return_2_values(training_dataset),
                                      validation_data=return_2_values(validation_dataset), steps_per_epoch=TRAIN_STEPS,
                                      epochs=EPOCHS_FINE_TUNE, callbacks=callbacks)

        model = set_backbone_trainable(model, metrics, True, CONFIG)

        history = model.fit(return_2_values(training_dataset), validation_data=return_2_values(validation_dataset),
                            steps_per_epoch=TRAIN_STEPS, initial_epoch=EPOCHS_FINE_TUNE, epochs=EPOCHS_FULL, callbacks=callbacks)

        history = join_history(history_fine_tune, history)
        print(history.history)

        model.save(model_file_path)

        if CONFIG.use_metrics:
            display_training_curves(history.history['auc'][1:], history.history['val_auc'][1:], 'auc', 211)
        display_training_curves(history.history['loss'][1:], history.history['val_loss'][1:], 'loss', 212)
        plt.savefig(os.path.join(CONFIG.work_dir, f'loss{fold}.png'))

        validation_with_augm_dataset = get_validation_dataset(val_filenames_folds[fold], is_augment=False)
        #validation_with_augm_dataset = validation_with_augm_dataset.repeat()
        validation_with_augm_dataset_tta = None
        steps = TRAIN_STEPS * 3
        submission.calc_and_save_submissions(CONFIG, model, f'with_augm_val_{fold}',
                                             #(validation_with_augm_dataset, steps), validation_with_augm_dataset_tta,
                                             validation_with_augm_dataset, validation_with_augm_dataset_tta,
                                             ttas=0)
        del validation_with_augm_dataset
        del validation_with_augm_dataset_tta

        validation_dataset = get_validation_dataset(val_filenames_folds[fold])
        validation_dataset_tta = get_validation_dataset_tta(val_filenames_folds[fold])
        submission.calc_and_save_submissions(CONFIG, model, f'val_{fold}', validation_dataset, validation_dataset_tta,
                                             CONFIG.ttas)
        del validation_dataset
        del validation_dataset_tta


        test_dataset = get_test_dataset_with_labels(test_filenames)
        test_dataset_tta = get_test_dataset_with_labels_tta(test_filenames)
        submission.calc_and_save_submissions(CONFIG, model, f'test_{fold}', test_dataset, test_dataset_tta, CONFIG.ttas)

        del test_dataset
        del test_dataset_tta

        if CONFIG.save_last_epochs!=0:
            models=[]
            filepaths=save_callback.get_filepaths()
            for filepath in filepaths:
                m=tf.keras.models.load_model(filepath, custom_objects={'BinaryFocalLoss':BinaryFocalLoss}, compile=True)
                m.trainable=False
                models.append(m)
            submission.calc_and_save_submissions(CONFIG, models, f'val_le_{fold}', validation_dataset,
                                                 validation_dataset_tta,
                                                 CONFIG.ttas)
            submission.calc_and_save_submissions(CONFIG, models, f'test_le_{fold}', test_dataset,
                                                 test_dataset_tta, CONFIG.ttas)

    if (not is_local) and (not is_kaggle):
        if fold!=0:
            subprocess.check_call(['gsutil', 'rm', '-r', CONFIG.gs_work_dir])
        subprocess.check_call(['gsutil', '-m', 'cp', '-r', CONFIG.work_dir,CONFIG.gs_work_dir])

    gc.collect()

def calc_mean_auc(subm_pattern):
    subms = glob.glob(os.path.join(CONFIG.work_dir, subm_pattern))
    subms = [pd.read_csv(s) for s in subms]
    auc = np.mean([calc_auc(s) for s in subms])
    return auc


val_auc=calc_mean_auc('val_*_tta_*.csv')

with_augm_auc=calc_mean_auc('with_augm_val_*.csv')

test_auc=calc_mean_auc('test_*_tta_*.csv')

test_subms=glob.glob(os.path.join(CONFIG.work_dir,'test_*_tta_*.csv'))
test_subms=[pd.read_csv(s) for s in test_subms]
avg_test_sub=submission.avg_submissions(test_subms)
test_avg_auc=calc_auc(avg_test_sub)

with open(os.path.join(CONFIG.work_dir,'metric.txt'),'wt') as file:
    text=f'val_auc,with_augm_auc,test_auc,avg_test_auc\n{val_auc},{with_augm_auc},{test_auc},{test_avg_auc}'
    print(text)
    file.write(text)