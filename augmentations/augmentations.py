import tensorflow as tf
import tensorflow.keras.backend as K
import math
from augmentations.augmentation_hair import hair_aug_tf
from augmentations.augmentation_microscope import microscope_aug_tf
from augmentations.augmentations_geom import transform_geometricaly
from config.consts import *
from efficientnet.keras import preprocess_input
# COARSE DROPOUT

def cutout(image, image_height=256, prob = 0.75, holes_count = 8, hole_size = 0.2):
    # input image - is one image of size [image_height,image_height,3] not a batch of [b,image_height,image_height,3]
    # output - image with CT squares of side size SZ*image_height removed
    
    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([],0,1, seed=op_seed) < prob, tf.int32)
    if (P==0)|(holes_count == 0)|(hole_size == 0):
        return image
    
    for k in range(holes_count):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,image_height, seed=op_seed),tf.int32)
        y = tf.cast( tf.random.uniform([],0,image_height, seed=op_seed),tf.int32)
        # COMPUTE SQUARE
        WIDTH = tf.cast(hole_size * image_height, tf.int32)
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(image_height,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(image_height,x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        two = tf.ones([yb-ya,xb-xa,3])*tf.random.uniform((),minval=-1, maxval=1.0)
        three = image[ya:yb,xb:image_height,:]
        middle = tf.concat([one,two,three],axis=1)
        image = tf.concat([image[0:ya,:,:],middle,image[yb:image_height,:,:]],axis=0)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    image = tf.reshape(image,[image_height,image_height,3])
    return image


def _augment_color(image):
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.05)

    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)

    return image


def cut_mix(batch,labels, *args, config, prob=0.5):
    images = batch['image']
    symbolic_shape = K.shape(images)
    batch_size = symbolic_shape[0]
    if (prob==0) or (batch_size!=config.batch_size):
        return batch,labels

    images_augm = []
    labels_augm = []


    for i in range(config.batch_size):

        img, lab = images[i],labels[i]

        cut_area_percent=tf.random.uniform((),0,1.0)
        if tf.random.uniform((), 0, 1.0)>prob:
            cut_area_percent=0.0

        cut_area=cut_area_percent*config.image_height*config.image_height
        cut_size=tf.cast(tf.sqrt(cut_area),tf.int32)

        dx=tf.cast((config.image_height-cut_size),tf.float32)
        x = tf.cast(tf.random.uniform((),0,1)*dx,tf.int32)
        xmax=tf.minimum(config.image_height,x+cut_size)
        dy=tf.cast((config.image_height - cut_size),tf.float32)
        y = tf.cast(tf.random.uniform((), 0, 1) * dy, tf.int32)
        ymax = tf.minimum(config.image_height, y + cut_size)
        donor_idx = tf.cast( tf.random.uniform([],0,config.batch_size),tf.int32)
        donor=images[donor_idx]

        one = img[y:ymax, 0:x, :]
        two = donor[y:ymax,x:xmax]
        three = img[y:ymax, xmax:]
        middle = tf.concat([one, two, three], axis=1)
        mixed = tf.concat([img[0:y, :, :], middle, img[ymax:]], axis=0)
        mixed = tf.reshape(mixed, (config.image_height, config.image_height, 3))

        label_mixed=lab*(1-cut_area_percent)+labels[donor_idx]*cut_area_percent
        images_augm.append(mixed)
        labels_augm.append(label_mixed)



    images=tf.stack(images_augm,axis=0)
    labels = tf.stack(labels_augm, axis=0)
    images = tf.reshape(images, (len(images), config.image_height, config.image_height, 3))
    labels = tf.reshape(labels, (len(labels),))

    batch['image']=images
    return batch,labels


def _normalize(image8u):
    image = tf.cast(image8u,tf.float32)
    image = preprocess_input(image)
    return image


def augment_train(input_dict, labels, config):
    image=input_dict['image']
    image=_augment_color(image)
    image = hair_aug_tf(image, config)
    image = microscope_aug_tf(image, config)

    image = _normalize(image)
    image = cutout(image, config.image_height, config.cut_out_prob)

    image = transform_geometricaly(image, DIM=config.image_height)
    input_dict['image']=image
    return input_dict, labels


def augment_tta(input_dict, labels, config):
    image=input_dict['image']
    image=_augment_color(image)
    image = _normalize(image)
    image = transform_geometricaly(image, DIM=config.image_height)
    input_dict['image'] = image
    return input_dict, labels

def augment_val(input_dict, labels, config):
    image=input_dict['image']
    image = _normalize(image)
    input_dict['image'] = image
    return input_dict, labels

def augment_val_aug(input_dict, labels, config):
    image=input_dict['image']
    image = _normalize(image)
    #image = tf.image.random_flip_left_right(image)
    input_dict['image'] = image
    return input_dict, labels

def augment_test(input_dict, labels, config):
    image=input_dict['image']
    image = _normalize(image)
    input_dict['image'] = image
    return input_dict, labels
