import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
from dataset.dict_list import DictList
from model.create_model import load_model
from datetime import datetime


from dataset.display_utils import get_high_low_loss_images

def _batch_to_tensor(batch):
    return {k: tf.convert_to_tensor(np.expand_dims(v, axis=0)) for k, v in batch.items()}

def calc_occlustion_map(output_modifier_fn, model, batch, offset, steps, config, average_samples=10):
    image = batch['image']
    h, w = image.shape[:2]
    dh = max(1, h // steps)
    dw = max(1, w // steps)

    start=datetime.now()
    batch_list=[]
    del batch['image_name'] #TPU does not support string data type in datasets
    for y in range(0, h, dh):
        for x in range(0, w, dw):
            for _ in range(average_samples):
                batch_copy = batch.copy()
                image_copy = copy.deepcopy(image)
                patch = image_copy[y:y + dh, x:x + dw]
                pixels = np.reshape(patch, (-1, 3))
                np.random.shuffle(pixels)
                pixels = np.reshape(pixels, patch.shape)
                image_copy[y:y + dh, x:x + dw] = pixels
                batch_copy['image'] = image_copy
                batch_list.append(batch_copy)

    batch_list=DictList(batch_list)

    finish = datetime.now()
    print(f'build batch_list ={finish-start}')

    start=finish
    dataset=tf.data.Dataset.from_tensor_slices(batch_list.dict)
    dataset=dataset.batch(config.batch_size_inference)
    predictions=model.predict(dataset)
    #print(predictions[:10])
    finish = datetime.now()
    print(f'predict  ={finish - start}')

    start=finish
    occl_list = [offset - output_modifier_fn(p) for p in predictions]

    occlusion_map=[]
    idx=0
    for y in range(0, h, dh):
        row=[]
        for x in range(0, w, dw):
            occl = 0
            for _ in range(average_samples):
                occl+=occl_list[idx]
                idx+=1
            row.append(occl/average_samples)
        occlusion_map.append(np.array(row))

    occlusion_map = np.array(occlusion_map).astype(np.float32)
    occlusion_map = cv2.resize(occlusion_map, (w, h))
    finish=datetime.now()
    print(f'reconstruct map  ={finish - start}')
    return occlusion_map

def calc_occlusion_map_multisace(output_modifier_fn, model, batch, config, steps_list=(10,5,3), average_samples=10):

    offset=output_modifier_fn(model(_batch_to_tensor(batch)).numpy()[0])

    occlusion_map=0
    for steps in steps_list:
        single_scal_saliency_map=calc_occlustion_map(output_modifier_fn, model, batch, offset, steps, config,  average_samples)
        perturbation_area=1/(steps*steps)
        occlusion_map+=single_scal_saliency_map/perturbation_area

    occlusion_map/=len(steps_list)
    return occlusion_map


def normalize(img,to_gray=False):
    rgb=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if to_gray:
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    else:
        return rgb

def save_interpretations(model,test_dataset,dst_dir, config, average_samples=10, steps_list=(3,), N=3):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    subplot = [2,N,1]

    def loss_fn(images, labels):
        outputs = model(images)
        losses = [model.loss([l],[o]) for o, l in zip(outputs, labels)]
        losses = tf.stack(losses, axis=0).numpy().astype(np.float32)
        return losses

    start=datetime.now()
    print(f'start get_high_low_loss_images')
    (high_loss_images, high_loss_labels, high_loss_loss), (low_loss_images, low_loss_labels, low_loss_loss) = get_high_low_loss_images(test_dataset, N, loss_fn, max_batches=None)
    finish=datetime.now()
    print(f'finished get_high_low_loss_images {finish-start}')
    plt.figure(figsize=(13, 13))

    batches=high_loss_images+low_loss_images
    labels=high_loss_labels+low_loss_labels
    losses = high_loss_loss + low_loss_loss

    def output_fn(output):
        return output[1]

    saliency_maps = [calc_occlusion_map_multisace(output_fn, model, batch,config, average_samples=average_samples, steps_list=steps_list) for batch in batches]

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
    import timeit

    model=load_model('/mnt/850G/GIT/kagle_melanoma_tf_tpu/artifacts/trained_models/model0_0.h5')
    validation_dataset_tta = get_test_dataset(['/mnt/850G/GIT/kagle_melanoma_tf_tpu/data/256x256_triple/train00-2182.tfrec','/mnt/850G/GIT/kagle_melanoma_tf_tpu/data/256x256_triple/train01-2185.tfrec'], CONFIG)

    def stm():
        save_interpretations(model, validation_dataset_tta, CONFIG.work_dir, CONFIG)#, average_samples=1, steps_list=(10,),N=1)
    print(timeit.timeit(stm,number=1))
