import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
from model.create_model import load_model

from dataset.display_utils import get_high_low_loss_images, display_one_image

def _batch_to_tensor(batch):
    return {k: tf.convert_to_tensor(np.expand_dims(v, axis=0)) for k, v in batch.items()}

def calc_occlusion_map(output_modifier_fn, model, batch, steps=10, occlusion_color=1.0):
    image=batch['image']
    h,w=image.shape[:2]
    dh = max(1, h//steps)
    dw = max(1, w // steps)
    offset=output_modifier_fn(model(_batch_to_tensor(batch)))
    occlusion_map=[]
    for x in range(0,w,dw):
        row=[]
        for y in range(0,h,dh):
            batch_copy=batch.copy()
            image_copy=copy.deepcopy(image)
            image_copy[y:y+dh,x:x+dw]=occlusion_color
            batch_copy['image']=image_copy
            occl=output_modifier_fn(model(_batch_to_tensor(batch_copy)))-offset
            row.append(occl)
        occlusion_map.append(np.array(row))

    occlusion_map=np.array(occlusion_map).astype(np.float32)
    occlusion_map=cv2.resize(occlusion_map,(w,h))
    return occlusion_map


def normalize(img,to_gray=False):
    rgb=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if to_gray:
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    else:
        return rgb

def save_interpretations(model,test_dataset,dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    N=3
    subplot = [2,3,1]

    def loss_fn(images, labels):
        outputs = model(images)
        losses = [model.loss([l],[o]) for o, l in zip(outputs, labels)]
        losses = tf.concat(losses, axis=0)
        return losses

    (high_loss_images, high_loss_labels, high_loss_loss), (low_loss_images, low_loss_labels, low_loss_loss) = get_high_low_loss_images(test_dataset, N, loss_fn, max_batches=None)


    plt.figure(figsize=(13, 13))

    batches=high_loss_images+low_loss_images
    labels=high_loss_labels+low_loss_labels
    losses = high_loss_loss + low_loss_loss

    def output_fn(output):
        return output[:,1].numpy()[0]

    saliency_maps = [calc_occlusion_map(output_fn, model, batch,
                            steps=10) for batch in batches]

    red=True
    for i, batch in enumerate(batches):

        title = str(labels[i])+' '+str(round(losses[i],3))
        if i>=N:
            red=False

        saliency_map = normalize(saliency_maps[i])
        plt.subplot(*subplot)
        plt.axis('off')
        image=batch['image']
        plt.imshow(normalize(image,to_gray=True))
        plt.imshow(saliency_map, cmap='jet', alpha=0.2)
        plt.title(title, fontsize=16, color='red' if red else 'black')
        subplot[2]+=1


    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(os.path.join(dst_dir,'high_low_loss.png'))

if __name__=="__main__":
    from dataset.dataset_utils import *
    from config.config import CONFIG
    from config.runtime import get_scope

    scope=get_scope()
    with scope:
        model=load_model('/mnt/850G/GIT/kagle_melanoma_tf_tpu/artifacts/trained_models/model0.h5')
        validation_dataset_tta = get_test_dataset(['/mnt/850G/GIT/kagle_melanoma_tf_tpu/data/128x128/train00-2071.tfrec'], CONFIG)

        save_interpretations(model, validation_dataset_tta, CONFIG.work_dir)