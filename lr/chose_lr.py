import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
print("Tensorflow version " + tf.__version__)
from dataset.dataset_utils import *
from model.create_model import create_model

train_filenames_folds, val_filenames_folds =get_train_val_filenames(DATASETS[IMAGE_HEIGHT]['new'] ,CONFIG.nfolds)
lrs = np.linspace(1e-6, 1e-5, 10)
folds=1
val_metrics = []

for lr in tqdm(lrs, desc='lr'):
    final_metrics = []
    for fold in range(folds):

        print(f'fold={fold}')

        metrics = [tf.keras.metrics.AUC(name='auc')]
        model = create_model(CONFIG, metrics, backbone_trainable=True, lr=lr)

        training_dataset = get_training_dataset(train_filenames_folds[fold], DATASETS[IMAGE_HEIGHT]['old'])

        if TRAIN_STEPS is None:
            TRAIN_STEPS = count_data_items(train_filenames_folds[fold] )//BATCH_SIZE

        print(f'TRAIN_STEPS={TRAIN_STEPS}')
        validation_dataset = get_validation_dataset(val_filenames_folds[fold])

        history = model.fit(return_2_values(training_dataset),
                                          validation_data=return_2_values(validation_dataset), steps_per_epoch=TRAIN_STEPS,
                                          epochs=EPOCHS_FULL, callbacks=None)

        final_metric = history.history["val_auc"][-1]
        final_metrics.append(final_metric)
    final_metrics = np.mean(final_metrics)
    print(f'lr {lr} metric={final_metrics}')
    val_metrics.append(final_metrics)

plt.plot(lrs, val_metrics)
plt.show()



