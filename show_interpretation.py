import tensorflow as tf
from display_utils import display_9_images_from_dataset
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from dataset_utils import get_train_val_filenames, get_training_dataset, return_2_values
from display_utils import dataset_to_numpy_util
from consts import CONFIG, IMAGE_HEIGHT, DATASETS
from create_model import create_model

if __name__=="__main__":

    num_imgs=9
    show_zero_labels=False
    model_path='/home/sergey/GIT/kagle_melanoma_tf_tpu_0/artifacts/val_quality_B1_bce_loss_128_epochs_8_cut_mix_0/model0.h5'
    model = create_model(CONFIG, metrics=None, backbone_trainable=False)
    model = tf.keras.models.load_model(model_path, compile=True)
    model.trainable=False

    fold=0
    train_files_folds, val_files_folds = get_train_val_filenames(DATASETS[IMAGE_HEIGHT]['new'],CONFIG.nfolds)

    dataset = get_training_dataset([train_files_folds[fold][0]], DATASETS[IMAGE_HEIGHT]['old'],CONFIG)

    images, labels = dataset_to_numpy_util(return_2_values(dataset), num_imgs, show_zero_labels)
    images=np.array(images)
    image_titles=[str(l) for l in labels]

    def loss(output):
        return output

    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m

    # Create Saliency object.
    # If `clone` is True(default), the `model` will be cloned,
    # so the `model` instance will be NOT modified, but it takes a machine resources.
    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)

    # Generate saliency map
    saliency_map = saliency(loss, images)
    saliency_map = normalize(saliency_map)

    ncols=int(np.sqrt(num_imgs))
    nrows=num_imgs//ncols
    subplot_args = {'nrows': nrows, 'ncols': ncols, 'figsize': (9, 3),
                    'subplot_kw': {'xticks': [], 'yticks': []}}

    i=0
    plt.figure(figsize=(13, 13))
    for row in range(nrows):
        for col in range(ncols):
            plt.subplot(nrows,ncols,i+1)
            plt.title(image_titles[i], fontsize=14)
            plt.axis('off')
            plt.imshow(saliency_map[i], cmap='jet')

            i+=1

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.tight_layout()
    plt.show()