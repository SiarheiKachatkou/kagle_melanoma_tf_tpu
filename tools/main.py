import os
import pickle
import gc
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import subprocess
import config.config
from matplotlib import pyplot as plt
from model.lr import get_lrfn, get_cycling_lrfn, get_lrfn_fine_tune
from dataset.display_utils import display_training_curves, plot_lr
from config.config import CONFIG, TRAIN_STEPS,  metrics_path
from dataset.dataset_utils import *
from config.consts import DATASETS
from submission import submission
import shutil
from model.create_model import BinaryFocalLoss
from model.SaveBestNCallback import SaveBestNCallback
from model.create_model import create_model, set_backbone_trainable, load_model
from config.runtime import get_scope
from model.sparceauc import SparceAUC
from submission import submit
from model.history import join_history

if not os.path.exists(CONFIG.work_dir):
    os.makedirs(CONFIG.work_dir)

with open(os.path.join(CONFIG.work_dir,'config.yaml'),'wt') as file:
    config_dict=dict(CONFIG._asdict())
    file.write(yaml.dump(config_dict))

lrfn = eval(CONFIG.lr_fn)
plot_lr(lrfn,CONFIG.epochs_full,CONFIG.work_dir)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

train_filenames_folds, val_filenames_folds, test_val_filenames = get_train_val_filenames(DATASETS[CONFIG.image_height]['new'],CONFIG.nfolds)
test_filenames=get_test_filenames(DATASETS[CONFIG.image_height]['new'])
if is_debug:
    test_filenames = [test_filenames[0]]
    train_filenames_folds=[[f[0:2]] for f in train_filenames_folds]
    val_filenames_folds=[[f[0:2]] for f in val_filenames_folds]

preds_fold_avg=[]

for fold in range(CONFIG.nfolds):

    print(f'fold={fold}')
    model_dir_path=f'{CONFIG.work_dir}/../trained_models/'
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    model_file_path=os.path.join(model_dir_path,f'model{fold}.h5')

    callbacks=[lr_callback]
    validation_dataset_tta = get_validation_dataset_tta(val_filenames_folds[fold], CONFIG)
    if CONFIG.save_last_epochs != 0:
        save_callback_best_n = SaveBestNCallback(CONFIG.work_dir, fold, CONFIG.save_last_epochs, metric_name='val_auc', mode='max',
                                               val_ttas=CONFIG.val_ttas, val_dataset=validation_dataset_tta)
        callbacks.append(save_callback_best_n)

    scope = get_scope()
    with scope:
        metrics = [SparceAUC(name="auc"),tf.keras.metrics.SparseCategoricalCrossentropy(name='loss_no_reg')] if CONFIG.use_metrics else None

        opt = tf.keras.optimizers.Adam(learning_rate=CONFIG.lr_start)

        model = create_model(CONFIG, metrics, optimizer=opt, backbone_trainable=False)

        model.summary()
        train_filenames_old = tf.io.gfile.glob(DATASETS[CONFIG.image_height]['old'])
        training_dataset = get_training_dataset(train_filenames_folds[fold], train_filenames_old, CONFIG)
        if TRAIN_STEPS is None:
            TRAIN_STEPS=(count_data_items(train_filenames_folds[fold])+count_data_items(train_filenames_old))//CONFIG.batch_size

        print(f'TRAIN_STEPS={TRAIN_STEPS}')
        validation_dataset = get_validation_dataset(val_filenames_folds[fold],CONFIG)

        history_fine_tune = model.fit(remove_str(training_dataset),
                                      validation_data=remove_str(validation_dataset) if do_validate else None,
                                      steps_per_epoch=TRAIN_STEPS,
                                      epochs=CONFIG.epochs_fine_tune, callbacks=[lr_callback])

        model = set_backbone_trainable(model, metrics, optimizer=opt, flag=True, cfg=CONFIG, fine_tune_last=CONFIG.fine_tune_last)

        save_callback_best = ModelCheckpoint(
            filepath=model_file_path, monitor='val_auc', verbose=1, save_best_only=True,
            mode='max', save_freq='epoch')

        callbacks.append(save_callback_best)
        history = model.fit(remove_str(training_dataset),
                            validation_data=remove_str(validation_dataset)  if do_validate else None,
                            steps_per_epoch=TRAIN_STEPS, initial_epoch=CONFIG.epochs_fine_tune, epochs=CONFIG.epochs_full, callbacks=callbacks)
        history=join_history(history_fine_tune,history)
        print(history.history)

        if do_validate:
            final_auc = history.history["val_auc"][-5:]
            print("FINAL VAL AUC MEAN-5: ", np.mean(final_auc))
            if CONFIG.use_metrics:
                display_training_curves(history.history['auc'][1:], history.history['val_auc'][1:], 'auc', 211)
            display_training_curves(history.history['loss_no_reg'][1:], history.history['val_loss_no_reg'][1:], 'loss', 212)
            plt.savefig(os.path.join(CONFIG.work_dir, f'loss{fold}.png'))

        model=load_model(model_file_path)

        subms=submission.make_submission_dataframe(get_validation_dataset_tta(test_val_filenames,CONFIG), model, repeats=CONFIG.ttas)
        preds_fold_avg.append(submission.avg_submissions(subms))

        validation_dataset = get_validation_dataset(val_filenames_folds[fold],CONFIG)

        submission.calc_and_save_submissions(CONFIG, model, f'val_{fold}', validation_dataset, validation_dataset_tta,
                                             CONFIG.ttas)
        del validation_dataset
        del validation_dataset_tta

        test_dataset = get_test_dataset(test_filenames,CONFIG)
        test_dataset_tta = get_test_dataset_tta(test_filenames,CONFIG)
        submission.calc_and_save_submissions(CONFIG, model, f'test_{fold}', test_dataset, test_dataset_tta, CONFIG.ttas)
        del test_dataset
        del test_dataset_tta

        if CONFIG.save_last_epochs!=0:
            models=[]
            filepaths=save_callback_best.get_filepaths()
            for filepath in filepaths:
                m=load_model(filepath)
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

val_avg_tta_le_auc, val_avg_tta_auc = submit.main(CONFIG.nfolds,CONFIG.work_dir)
test_avg_tta_auc = submit.calc_auc(submission.avg_submissions(preds_fold_avg))

metric_dir=os.path.dirname(metrics_path)
if not os.path.exists(metric_dir):
    os.mkdir(metric_dir)
with open(metrics_path,'wt') as file:
    file.write(f'val_avg_tta_le_auc:\n    {val_avg_tta_le_auc}\nval_avg_tta_auc:\n    {val_avg_tta_auc}\ntest_avg_tta_auc:\n    {test_avg_tta_auc}')