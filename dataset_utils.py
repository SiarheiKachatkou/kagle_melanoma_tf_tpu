import re
from sklearn.model_selection import KFold
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
from consts import *

def dataset_to_numpy_util(dataset, N):
    dataset = dataset.unbatch().batch(N)
    for images, labels in dataset:
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        break;  
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    label = np.argmax(label, axis=-1)  # one-hot to class number
    correct_label = np.argmax(correct_label, axis=-1) # one-hot to class number
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], str(correct), ', shoud be ' if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=16, color='red' if red else 'black')
    return subplot+1
  
def display_9_images_from_dataset(dataset):
    subplot=331
    plt.figure(figsize=(13,13))
    #labels=[0,0]
    #while sum(labels)==0:
    images, labels = dataset_to_numpy_util(dataset, 9)
    labels=np.argmax(labels, axis=-1)
    print(labels)
    for i, image in enumerate(images):
        title = CLASSES[labels[i]]
        subplot = display_one_flower(image, title, subplot)
        if i >= 8:
            break;
              
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()  

def display_9_images_with_predictions(images, predictions, labels):
    subplot=331
    plt.figure(figsize=(13,13))
    for i, image in enumerate(images):
        title, correct = title_from_label_and_target(predictions[i], labels[i])
        subplot = display_one_flower(image, title, subplot, not correct)
        if i >= 8:
            break;
              
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        #plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
    
    

features_test = {
      'image': tf.io.FixedLenFeature([], tf.string),
      'image_name': tf.io.FixedLenFeature([], tf.string),
      'patient_id': tf.io.FixedLenFeature([], tf.int64),
      'sex': tf.io.FixedLenFeature([], tf.int64),
      'age_approx': tf.io.FixedLenFeature([], tf.int64),
      'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64)
  }


features = features_test.copy()
features['target']=tf.io.FixedLenFeature([], tf.int64)
  

def read_tfrecord(example, is_test):
    
    the_features=features_test if is_test else features
    
    example = tf.io.parse_single_example(example, the_features)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0 
    image_name = tf.cast(example['image_name'], tf.string)
    if is_test:
        class_label = tf.constant(0,dtype=tf.int32)
    else:
        class_label = tf.cast(example['target'], tf.int32)
    one_hot_class_label=tf.one_hot(class_label, depth=len(CLASSES))
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
    the_read_tfrecord=partial(read_tfrecord,is_test=is_test)
    dataset = dataset.map(the_read_tfrecord, num_parallel_calls=AUTO)
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
    #image,one_hot_class = albumentaze_data(image,one_hot_class,IMAGE_SIZE)
    return image, one_hot_class, image_name

def get_training_dataset(training_fileimages):
    dataset = load_dataset(training_fileimages, is_test=False)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    #dataset = dataset.map(partial(albumentaze_data, img_size=IMAGE_SIZE), num_parallel_calls=AUTO)
    
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    
    dataset = dataset.batch(BATCH_SIZE)
    
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(validation_fileimages):
    dataset = load_dataset(validation_fileimages, is_test=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def data_tta(image, one_hot_class, image_name):
    
    image = tf.image.random_saturation(image, 0, 2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image,0.8,1.2)
    return image, one_hot_class, image_name

def get_validation_dataset_tta(val_filenames):
    dataset = load_dataset(val_filenames, is_test=False)
    dataset = dataset.map(data_tta, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

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