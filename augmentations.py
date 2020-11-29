import tensorflow as tf
import tensorflow.keras.backend as K
import math
from augmentation_hair import hair_aug_tf
from consts import *

# COARSE DROPOUT

def cutout(image, IMG_HEIGHT=256, prob = 0.75, holes_count = 8, hole_size = 0.2):
    # input image - is one image of size [IMG_HEIGHT,IMG_HEIGHT,3] not a batch of [b,IMG_HEIGHT,IMG_HEIGHT,3]
    # output - image with CT squares of side size SZ*IMG_HEIGHT removed
    
    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([],0,1, seed=op_seed) > prob, tf.int32)
    if (P==0)|(holes_count == 0)|(hole_size == 0):
        return image
    
    for k in range(holes_count):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,IMG_HEIGHT, seed=op_seed),tf.int32)
        y = tf.cast( tf.random.uniform([],0,IMG_HEIGHT, seed=op_seed),tf.int32)
        # COMPUTE SQUARE
        WIDTH = tf.cast(hole_size * IMG_HEIGHT, tf.int32)
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(IMG_HEIGHT,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(IMG_HEIGHT,x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        two = tf.ones([yb-ya,xb-xa,3])*tf.random.uniform((),minval=-1, maxval=1.0)
        three = image[ya:yb,xb:IMG_HEIGHT,:]
        middle = tf.concat([one,two,three],axis=1)
        image = tf.concat([image[0:ya,:,:],middle,image[yb:IMG_HEIGHT,:,:]],axis=0)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    image = tf.reshape(image,[IMG_HEIGHT,IMG_HEIGHT,3])
    return image

ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    rotation_matrix = get_3x3_mat([c1, s1, zero,
                                   -s1, c1, zero,
                                   zero, zero, one])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one, s2, zero,
                                zero, c2, zero,
                                zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one / height_zoom, zero, zero,
                               zero, one / width_zoom, zero,
                               zero, zero, one])
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one, zero, height_shift,
                                zero, one, width_shift,
                                zero, zero, one])

    return K.dot(K.dot(rotation_matrix, shear_matrix),
                 K.dot(zoom_matrix, shift_matrix))


def transform_geometricaly(image, DIM=256):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted

    image = tf.image.random_flip_left_right(image)

    XDIM = DIM % 2  # fix for size 331
    rot = ROT_ * tf.random.normal([1], seed=op_seed, dtype='float32')
    shr = SHR_ * tf.random.normal([1], seed=op_seed, dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], seed=op_seed, dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], seed=op_seed, dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], seed=op_seed, dtype='float32')
    w_shift = WSHIFT_ * tf.random.normal([1], seed=op_seed, dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])


def _augment_color(image):
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.05)

    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)

    return image


def cut_mix(images,labels,*args, prob=0.5):

    symbolic_shape = K.shape(images)
    batch_size = symbolic_shape[0]
    if (prob==0) or (batch_size!=BATCH_SIZE):
        return (images,labels,*args)

    images_augm = []
    labels_augm = []


    for i in range(BATCH_SIZE):

        img, lab = images[i],labels[i]

        cut_area_percent=tf.random.uniform((),0,1.0)
        if tf.random.uniform((), 0, 1.0)>prob:
            cut_area_percent=0.0

        cut_area=cut_area_percent*IMAGE_HEIGHT*IMAGE_HEIGHT
        cut_size=tf.cast(tf.sqrt(cut_area),tf.int32)

        dx=tf.cast((IMAGE_HEIGHT-cut_size),tf.float32)
        x = tf.cast(tf.random.uniform((),0,1)*dx,tf.int32)
        xmax=tf.minimum(IMAGE_HEIGHT,x+cut_size)
        dy=tf.cast((IMAGE_HEIGHT - cut_size),tf.float32)
        y = tf.cast(tf.random.uniform((), 0, 1) * dy, tf.int32)
        ymax = tf.minimum(IMAGE_HEIGHT, y + cut_size)
        donor_idx = tf.cast( tf.random.uniform([],0,BATCH_SIZE),tf.int32)
        donor=images[donor_idx]

        one = img[y:ymax, 0:x, :]
        two = donor[y:ymax,x:xmax]
        three = img[y:ymax, xmax:]
        middle = tf.concat([one, two, three], axis=1)
        mixed = tf.concat([img[0:y, :, :], middle, img[ymax:]], axis=0)
        mixed = tf.reshape(mixed, (IMAGE_HEIGHT, IMAGE_HEIGHT, 3))

        label_mixed=lab*(1-cut_area_percent)+labels[donor_idx]*cut_area_percent
        images_augm.append(mixed)
        labels_augm.append(label_mixed)



    images=tf.stack(images_augm,axis=0)
    labels = tf.stack(labels_augm, axis=0)
    images = tf.reshape(images, (len(images), IMAGE_HEIGHT, IMAGE_HEIGHT, 3))
    labels = tf.reshape(labels, (len(labels),))

    return (images,labels,*args)


def _normalize(image8u):
    image = tf.cast(image8u,tf.float32)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
    return image


def augment_train(image, label, image_name, config):
    image=_augment_color(image)
    image = _normalize(image)
    image = transform_geometricaly(image, DIM=IMAGE_HEIGHT)
    image = hair_aug_tf(image,config)
    return image, label, image_name


def augment_tta(image, label, image_name):
    image=_augment_color(image)
    image = _normalize(image)
    image = transform_geometricaly(image, DIM=IMAGE_HEIGHT)
    return image, label, image_name


def augment_val(image, label, image_name):
    image = _normalize(image)
    return image, label, image_name

def augment_val_aug(image, label, image_name):
    image = _normalize(image)
    #image = tf.image.random_flip_left_right(image)
    return image, label, image_name

def augment_test(image, label, image_name):
    image = _normalize(image)
    return image, label, image_name
