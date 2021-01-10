from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from config.consts import CLASSES
import cv2
import os
from dataset.dict_list import DictList
from tqdm import tqdm
from augmentations.augmentations import cut_mix

def dataset_to_numpy_util(dataset, N, show_zero_labels):
    dataset = dataset.unbatch().batch(N)
    chosen_images = []
    chosen_labels = []
    for input_dict, labels in dataset:

        images=input_dict['image']
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        for l, img in zip(numpy_labels,numpy_images):
            use_this_sample = False
            if show_zero_labels:
                if l==0:
                    use_this_sample = True
            else:
                if l!=0:
                    use_this_sample = True
            if use_this_sample:
                chosen_images.append(img)
                chosen_labels.append(l)

        if len(chosen_images)>=N:
            break
    return chosen_images,chosen_labels

def title_from_label_and_target(label, correct_label):
    label = np.argmax(label, axis=-1)  # one-hot to class number
    correct_label = np.argmax(correct_label, axis=-1) # one-hot to class number
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], str(correct), ', shoud be ' if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct


def display_one_image(image, title, subplot, red=False):
    if isinstance(subplot,str):
        plt.subplot(subplot)
    else:
        plt.subplot(*subplot)

    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=16, color='red' if red else 'black')
    if isinstance(subplot,str):
        return subplot + 1
    else:
        return [subplot[0],subplot[1],subplot[2]+1]


def get_high_low_loss_images(dataset, N, loss_fn, max_batches):

    batches_list = DictList()
    labels_list = []
    losses_list = []
    batch_count=0
    #TODO use model.predict(dataset) is is much faster
    for batch, labels in tqdm(dataset):

        losses_numpy=loss_fn(batch,labels)

        batches_list.extend(batch)
        labels_list.extend(labels.numpy())
        losses_list.extend(losses_numpy)

        batch_count+=1
        if max_batches is not None:
            if batch_count>max_batches:
                break
    args=np.argsort(losses_list)

    low_loss_batches = [batches_list[i] for i in args[:N]]
    low_loss_labels = [labels_list[i] for i in args[:N]]
    low_loss_loss = [losses_list[i] for i in args[:N]]

    args=args[::-1]

    high_loss_batches = [batches_list[i] for i in args[:N]]
    high_loss_labels = [labels_list[i] for i in args[:N]]
    high_loss_loss = [losses_list[i] for i in args[:N]]

    return (high_loss_batches, high_loss_labels, high_loss_loss), (low_loss_batches,low_loss_labels, low_loss_loss)


def display_9_images_from_dataset(dataset, show_zero_labels, loss_fn=None):
    subplot = 331
    plt.figure(figsize=(13, 13))

    if loss_fn is None:
        images, labels = dataset_to_numpy_util(dataset, 9, show_zero_labels)
    else:
        images, labels = get_high_low_loss_images(dataset, 9, loss_fn, max_batches=10)

    print(labels)
    for i, image in enumerate(images):
        image = cv2.normalize(image,None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        title = str(labels[i])
        subplot = display_one_image(image, title, subplot)
        if i >= 8:
            break

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def display_9_images_with_predictions(images, predictions, labels):
    subplot = 331
    plt.figure(figsize=(13, 13))
    for i, image in enumerate(images):
        title, correct = title_from_label_and_target(predictions[i], labels[i])
        subplot = display_one_image(image, title, subplot, not correct)
        if i >= 8:
            break;

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def display_training_curves(training, validation, title, subplot, validation_tta=None):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        # plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    if validation_tta is not None:
        ax.plot(validation_tta)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    # ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    legend=['train', 'valid']
    if validation_tta is not None:
        legend.append('valid_tta')
    ax.legend(legend)


def plot_lr(lrfn,epochs_full,work_dir):
    rng = [i for i in range(epochs_full)]
    y = [lrfn(x) for x in rng]
    plt.plot(rng, y)
    plt.title("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
    plt.savefig(os.path.join(work_dir, 'lr_schedule.png'))
