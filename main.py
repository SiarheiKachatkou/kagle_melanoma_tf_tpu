import gc
import yaml
import subprocess
from matplotlib import pyplot as plt
from debugtools.display_utils import display_training_curves, plot_lr
from config.getconfig import get_config
from dataset.dataset_utils import *
from dataset.files_utils import get_test_filenames,get_train_val_filenames, count_data_items
from config.consts import DATASETS
from submission import submission
from lr.lr import get_lrfn,get_cycling_lrfn
from model.create_model import BinaryFocalLoss
from model.savelastcallback import SaveLastCallback
from model.create_model import create_model, set_backbone_trainable
from config.runtime import get_scope
from model.history import join_history

CONFIG=get_config()
if not os.path.exists(CONFIG.work_dir):
    os.makedirs(CONFIG.work_dir)

with open(os.path.join(CONFIG.work_dir,'config.yaml'),'wt') as file:
    config_dict=dict(CONFIG._asdict())
    file.write(yaml.dump(config_dict))

lrfn = eval(CONFIG.lr_fn)
plot_lr(lrfn,CONFIG.epochs_full,CONFIG.work_dir)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

train_filenames_folds, val_filenames_folds=get_train_val_filenames(DATASETS[CONFIG.image_height]['new'],CONFIG.nfolds)
test_filenames=get_test_filenames(DATASETS[CONFIG.image_height]['new'])
if is_debug:
    test_filenames = [test_filenames[0]]
    train_filenames_folds=[[f[0]] for f in train_filenames_folds]
    val_filenames_folds=[[f[0]] for f in val_filenames_folds]

for fold in range(CONFIG.nfolds):

    print(f'fold={fold}')
    model_file_path=f'{CONFIG.work_dir}/model{fold}.h5'

    save_callback_best = tf.keras.callbacks.ModelCheckpoint(
        model_file_path, monitor='val_loss', verbose=0, save_best_only=True,
        mode='min', save_freq='epoch')
    save_callback_last=SaveLastCallback(CONFIG.work_dir, fold, CONFIG.epochs_full, CONFIG.save_last_epochs)

    save_callback=SaveLastCallback(CONFIG.work_dir,fold=fold, epochs=CONFIG.epochs_full,
                                   save_last_epochs=CONFIG.save_last_epochs)

    callbacks=[lr_callback,save_callback_best,save_callback_last]

    scope = get_scope()
    with scope:
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')] if CONFIG.use_metrics else None
        model = create_model(CONFIG, metrics, backbone_trainable=False)

        model.summary()
        training_dataset = get_training_dataset(train_filenames_folds[fold], DATASETS[CONFIG.image_height]['old'], CONFIG)
        if CONFIG.train_steps is None:
            train_steps=count_data_items(train_filenames_folds[fold])//CONFIG.batch_size
        else:
            train_steps=CONFIG.train_steps
        print(f'TRAIN_STEPS={train_steps}')
        validation_dataset = get_validation_dataset(val_filenames_folds[fold],CONFIG)

        history_fine_tune = model.fit(return_2_values(training_dataset),
                                      validation_data=return_2_values(validation_dataset) if do_validate else None,
                                      steps_per_epoch=train_steps,
                                      epochs=CONFIG.epochs_fine_tune, callbacks=callbacks)

        model = set_backbone_trainable(model, metrics, True, CONFIG)

        history = model.fit(return_2_values(training_dataset),
                            validation_data=return_2_values(validation_dataset)  if do_validate else None,
                            steps_per_epoch=train_steps, initial_epoch=CONFIG.epochs_fine_tune, epochs=CONFIG.epochs_full, callbacks=callbacks)

        history = join_history(history_fine_tune, history)
        print(history.history)

        model.save(model_file_path)
        if do_validate:
            final_accuracy = history.history["val_accuracy"][-5:]
            print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
            if CONFIG.use_metrics:
                display_training_curves(history.history['auc'][1:], history.history['val_auc'][1:], 'auc', 211)
            display_training_curves(history.history['loss'][1:], history.history['val_loss'][1:], 'loss', 212)
            plt.savefig(os.path.join(CONFIG.work_dir, f'loss{fold}.png'))



        validation_dataset = get_validation_dataset(val_filenames_folds[fold],CONFIG)
        validation_dataset_tta = get_validation_dataset_tta(val_filenames_folds[fold],CONFIG)
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
