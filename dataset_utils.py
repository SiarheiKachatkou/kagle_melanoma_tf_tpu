import re
from sklearn.model_selection import KFold
import numpy as np


import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
from consts import *
from augmentation_hair import hair_aug_tf
from augmentations import cutout, transform

features_test = {
      'image': tf.io.FixedLenFeature([], tf.string),
      'image_name': tf.io.FixedLenFeature([], tf.string),
      'patient_id': tf.io.FixedLenFeature([], tf.int64),
      'sex': tf.io.FixedLenFeature([], tf.int64),
      'age_approx': tf.io.FixedLenFeature([], tf.int64),
      'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64)
  }

features_old = {
  'image': tf.io.FixedLenFeature([], tf.string),
  'image_name': tf.io.FixedLenFeature([], tf.string),
  'patient_id': tf.io.FixedLenFeature([], tf.int64),
  'sex': tf.io.FixedLenFeature([], tf.int64),
  'age_approx': tf.io.FixedLenFeature([], tf.int64),
  'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
  'diagnosis': tf.io.FixedLenFeature([], tf.int64),
  'target': tf.io.FixedLenFeature([], tf.int64),
  'width': tf.io.FixedLenFeature([], tf.int64),
  'height': tf.io.FixedLenFeature([], tf.int64)
}

features = features_test.copy()
features['target']=tf.io.FixedLenFeature([], tf.int64)


def _normalize(image8u):
    image = tf.cast(image8u,tf.float32)
    image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')
    return image

  

def read_tfrecord(example):

    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = _normalize(image)
    image_name = tf.cast(example['image_name'], tf.string)

    class_label = tf.cast(example['target'], tf.int32)
    #one_hot_class_label=tf.one_hot(class_label, depth=len(CLASSES))
    one_hot_class_label = class_label
    return image, one_hot_class_label, image_name


def read_tfrecord_test(example):

    example = tf.io.parse_single_example(example, features_test)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = _normalize(image)
    image_name = tf.cast(example['image_name'], tf.string)

    class_label = tf.constant(0, dtype=tf.int32)

    one_hot_class_label = class_label# tf.one_hot(class_label, depth=len(CLASSES))
    return image, one_hot_class_label, image_name


def read_tfrecord_old(example):

    example = tf.io.parse_single_example(example, features_old)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = _normalize(image)
    image_name = tf.cast(example['image_name'], tf.string)

    class_label = tf.cast(example['target'], tf.int32)

    one_hot_class_label = class_label#tf.one_hot(class_label, depth=len(CLASSES))
    return image, one_hot_class_label, image_name


def force_image_sizes(dataset, image_size):
    # explicit size needed for TPU
    reshape_images = lambda image, *args: (tf.reshape(image, [*image_size, 3]), *args)
    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)
    return dataset


def load_dataset(filenames, is_test):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    the_read_tfrecord=read_tfrecord_test if is_test else read_tfrecord
    dataset = dataset.map(the_read_tfrecord, num_parallel_calls=AUTO)
    dataset = force_image_sizes(dataset, IMAGE_SIZE)
    return dataset


def load_dataset_old(fileimages_old):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(fileimages_old,
                                      num_parallel_reads=AUTO)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order)  # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_tfrecord_old, num_parallel_calls=AUTO)
    dataset = force_image_sizes(dataset, IMAGE_SIZE)
    return dataset

def data_augment(image, one_hot_class, image_name):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
   
    image = tf.image.random_saturation(image, 0, 2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image,0.8,1.2)
    image = cutout(image, IMAGE_HEIGHT, prob=0.5, holes_count=3, hole_size=0.2)
    shift=0.1*IMAGE_HEIGHT
    shear=0.01*IMAGE_HEIGHT
    zoom=0.1
    image = transform(image, IMAGE_HEIGHT, prob=0.5, rot_limit=180, shr_limit=shear,
                      hshift=shift, wshift=shift,
                      hzoom=zoom, wzoom=zoom)

    #image = hair_aug_tf(image, augment=True)
    #image,one_hot_class = albumentaze_data(image,one_hot_class,IMAGE_SIZE)
    return image, one_hot_class, image_name


def data_tta(image, one_hot_class, image_name):
    
    image = tf.image.random_saturation(image, 0, 2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image,0.8,1.2)
    return image, one_hot_class, image_name

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def get_test_filenames(gs_path_to_dataset_train):
    gcs_pattern = gs_path_to_dataset_train.replace('train','test')
    test_filenames = tf.io.gfile.glob(gcs_pattern)
    test_steps = count_data_items(test_filenames) // BATCH_SIZE
    print("TTEST IMAGES: ", count_data_items(test_filenames), ", STEPS PER EPOCH: ", test_steps)
    return test_filenames

def get_train_val_filenames(gs_path_to_dataset_train, nfolds):
    
    filenames = tf.io.gfile.glob(gs_path_to_dataset_train)
    filenames=np.array(filenames)
    kf = KFold(n_splits=nfolds, random_state=0, shuffle=True)
    train_filenames_folds=[]
    val_filenames_folds=[]
    for train_index, test_index in kf.split(filenames):
        train_filenames_folds.append(list(filenames[train_index]))
        val_filenames_folds.append(list(filenames[test_index]))
    return train_filenames_folds, val_filenames_folds


def get_training_dataset(training_fileimages, training_fileimages_old):
    dataset = load_dataset(training_fileimages, is_test=False)
    if len(training_fileimages_old)!=0:
        dataset_old = load_dataset_old(training_fileimages_old)
        dataset.concatenate(dataset_old)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def get_validation_dataset_tta(val_filenames):
    dataset = load_dataset(val_filenames, is_test=False)
    dataset = dataset.map(data_tta, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def get_validation_dataset(validation_fileimages):
    dataset = load_dataset(validation_fileimages, is_test=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset_tta(test_filenames):
    dataset = load_dataset(test_filenames, is_test=True)
    dataset = dataset.map(data_tta, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(test_filenames):
    dataset = load_dataset(test_filenames, is_test=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def return_2_values(dataset):
    def two(a1,a2,*args):
        return a1,a2
    ds=dataset.map(two,num_parallel_calls=AUTO)
    return ds