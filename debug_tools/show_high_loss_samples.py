import tensorflow as tf
from dataset.display_utils import display_9_images_from_dataset
from dataset.dataset_utils import get_train_val_filenames, get_training_dataset, return_2_values
from config.consts import CONFIG, IMAGE_HEIGHT, DATASETS
from model.create_model import create_model

if __name__=="__main__":

    model_path='/home/sergey/GIT/kagle_melanoma_tf_tpu_0/artifacts/val_quality_B1_bce_loss_128_epochs_8_cut_mix_0/model0.h5'
    model = create_model(CONFIG, metrics=None, backbone_trainable=False)
    model = tf.keras.models.load_model(model_path, compile=True)
    model.trainable=False

    loss = tf.keras.losses.BinaryCrossentropy(reduction='none')
    def loss_fn(images,labels):
        outputs=model(images)
        losses=[loss([o],[l]) for o,l in zip(outputs,labels)]
        losses=tf.concat(losses,axis=0)
        return losses

    fold=0
    train_files_folds, val_files_folds = get_train_val_filenames(DATASETS[IMAGE_HEIGHT]['new'],CONFIG.nfolds)

    dataset = get_training_dataset([train_files_folds[fold][0]], DATASETS[IMAGE_HEIGHT]['old'],CONFIG)
    display_9_images_from_dataset(return_2_values(dataset),show_zero_labels=None,loss_fn=loss_fn)