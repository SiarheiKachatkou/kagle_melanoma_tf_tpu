import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np


from dataset.display_utils import get_high_low_loss_images, display_one_image


def save_interpretations(model,test_dataset,dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    N=9
    subplot = 631

    def loss_fn(images, labels):
        outputs = model(images)
        losses = [model.loss([o], [l]) for o, l in zip(outputs, labels)]
        losses = tf.concat(losses, axis=0)
        return losses

    (high_loss_images, high_loss_labels, high_loss_loss), (low_loss_images, low_loss_labels, low_loss_loss) = get_high_low_loss_images(test_dataset, N, loss_fn, max_batches=None)


    plt.figure(figsize=(13, 13))

    images=high_loss_images+low_loss_images
    labels=high_loss_labels+low_loss_labels
    losses = high_loss_loss + low_loss_loss

    red=True
    for i, image in enumerate(images):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        title = str(labels[i])+' '+str(round(losses[i],3))
        if i>=N:
            red=False
        subplot = display_one_image(image, title, subplot,red)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(os.path.join(dst_dir,'high_low_loss.png'))

