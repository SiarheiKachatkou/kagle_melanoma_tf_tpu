import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
from consts import *
from functools import partial
#from augmentation_hair import hair_aug_tf
from augmentations import augment_train,augment_tta,augment_val,augment_test, cut_mix, augment_val_aug
from oversample import oversample
from files_utils import get_train_val_filenames,get_test_filenames,count_data_items

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

label_type=tf.float32

def read_tfrecord(example):

    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image_name = tf.cast(example['image_name'], tf.string)
    class_label = tf.cast(example['target'], label_type)
    return image, class_label, image_name


def read_tfrecord_wo_labels(example):

    example = tf.io.parse_single_example(example, features_test)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image_name = tf.cast(example['image_name'], tf.string)

    class_label = tf.constant(0, dtype=label_type)
    return image, class_label, image_name


def read_tfrecord_old(example):
    example = tf.io.parse_single_example(example, features_old)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image,dtype=tf.float32)
    image_name = tf.cast(example['image_name'], tf.string)
    class_label = tf.cast(example['target'], label_type)
    return image, class_label, image_name


def _num_parallel_calls():
    return 1 if is_debug else AUTO


def force_image_sizes(dataset, image_size):
    # explicit size needed for TPU
    reshape_images = lambda image, *args: (tf.reshape(image, [*image_size, 3]), *args)
    dataset = dataset.map(reshape_images, _num_parallel_calls())
    return dataset


def _ignore_order(dataset,is_deterministic):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = True
    if is_debug or is_deterministic:
        ignore_order.experimental_deterministic = True
    else:
        ignore_order.experimental_deterministic = False
    dataset = dataset.with_options(
        ignore_order)  # uses data as soon as it streams in, rather than in its original order
    return dataset


def load_dataset(filenames, is_wo_labels, is_deterministic, config):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=_num_parallel_calls()) # automatically interleaves reads from multiple files
    dataset = _ignore_order(dataset,is_deterministic)
    dataset = dataset.cache()
    the_read_tfrecord=read_tfrecord_wo_labels if is_wo_labels else read_tfrecord
    dataset = dataset.map(the_read_tfrecord, num_parallel_calls=_num_parallel_calls())
    dataset = force_image_sizes(dataset, [config.image_height,config.image_height])
    if (not is_wo_labels) and (not is_deterministic):
        dataset = dataset.shuffle(256)
        
    return dataset


def load_dataset_old(fileimages_old, config):

    dataset = tf.data.TFRecordDataset(fileimages_old,
                                      num_parallel_reads=_num_parallel_calls())  # automatically interleaves reads from multiple files
    dataset=_ignore_order(dataset,is_deterministic=False)
    dataset = dataset.cache()
    dataset = dataset.shuffle(512)
    dataset = dataset.map(read_tfrecord_old, num_parallel_calls=_num_parallel_calls())
    dataset = force_image_sizes(dataset, [config.image_height,config.image_height])
    return dataset

def _augm_dataset(dataset, augm_fn, batch_size):
    dataset = dataset.map(augm_fn, num_parallel_calls=_num_parallel_calls())
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

def _augm_batched_dataset(dataset,batch_augm_fn):
    dataset = dataset.map(batch_augm_fn, num_parallel_calls=_num_parallel_calls())
    return dataset


def _get_dataset(filenames,is_wo_labels,is_deterministic, augm_fn, batch_size, config):
    dataset = load_dataset(filenames, is_wo_labels=is_wo_labels,is_deterministic=is_deterministic, config=config)
    return _augm_dataset(dataset,augm_fn,batch_size)


def get_training_dataset(training_fileimages, training_fileimages_old, config, repeats=None):
    dataset = load_dataset(training_fileimages, is_wo_labels=False, is_deterministic=False, config=config)
    if len(training_fileimages_old)!=0:
        dataset_old = load_dataset_old(training_fileimages_old, config=config)
        dataset.concatenate(dataset_old)

    dataset = dataset.repeat(repeats)
    dataset=dataset.shuffle(512)
    if config.oversample_mult!=1:
        dataset=oversample(dataset,config)
    augm_fn=partial(augment_train,config=config)
    dataset = _augm_dataset(dataset,augm_fn,config.batch_size)
    return dataset

def get_validation_dataset_tta(val_filenames, config, cut_mix_prob=0):

    dataset = _get_dataset(val_filenames,is_wo_labels=False,augm_fn=partial(augment_tta,config=config),
                           batch_size=config.batch_size_inference,is_deterministic=True, config=config)
    if cut_mix_prob!=0:
        cut_mix_fn = partial(cut_mix, prob=cut_mix_prob)
        dataset = _augm_batched_dataset(dataset, cut_mix_fn)
    return dataset



def get_validation_dataset(val_filenames, config, is_augment=False):
    augm_fn=augment_val_aug if is_augment else  augment_val
    augm_fn = partial(augm_fn, config=config)
    return _get_dataset(val_filenames, is_wo_labels=False, augm_fn=augm_fn,batch_size=config.batch_size_inference,
                        is_deterministic=True, config=config)


def get_test_dataset_tta(test_filenames,config):
    return _get_dataset(test_filenames, is_wo_labels=True, augm_fn=partial(augment_tta, config=config),
                        batch_size=config.batch_size_inference,
                        is_deterministic=True, config=config)


def get_test_dataset(test_filenames,config):
    return _get_dataset(test_filenames, is_wo_labels=True, augm_fn=partial(augment_test,config=config),
                        batch_size=config.batch_size_inference,
                        is_deterministic=True, config=config)

def get_test_dataset_with_labels(test_filenames,config):
    return _get_dataset(test_filenames, is_wo_labels=False, augm_fn=partial(augment_test,config=config),
                        batch_size=config.batch_size_inference,
                        is_deterministic=True, config=config)

def get_test_dataset_with_labels_tta(test_filenames,config):
    return _get_dataset(test_filenames, is_wo_labels=False, augm_fn=partial(augment_tta,config=config)
                        ,batch_size=config.batch_size_inference,
                        is_deterministic=True, config=config)


def return_2_values(dataset):
    def two(a1,a2,*args):
        return a1,a2
    ds=dataset.map(two,num_parallel_calls=_num_parallel_calls())
    return ds