import tensorflow as tf
from config.consts import path_microscope_images
from augmentations.augmentations_geom import transform_geometricaly


microscope_images = tf.io.gfile.glob(path_microscope_images + '/*.png')
microscope_images_tf=tf.convert_to_tensor(microscope_images)

def _augm_color_microscope(microscope_img):
    microscope_img = tf.image.random_brightness(microscope_img, 10)

    microscope_img = tf.image.random_saturation(microscope_img, 0.95, 1.05)
    microscope_img = tf.image.random_hue(microscope_img, 0.05)

    microscope_img = tf.image.random_contrast(microscope_img, 0.5, 2.0)

    return microscope_img


def _augm_geom_microscope(microscope_img, image_height):
    microscope_img = transform_geometricaly(microscope_img, image_height)
    return microscope_img


def microscope_aug_tf(input_img_8u, config):

    if tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)<config.microscope_prob:

        # Copy the input image, so it won't be changed
        img = tf.identity(input_img_8u)

        # Read a random image
        i = tf.random.uniform(shape=[], maxval=tf.shape(microscope_images_tf)[0], dtype=tf.int32)
        fname = microscope_images_tf[i]
        bits = tf.io.read_file(fname)
        microscope_img = tf.image.decode_jpeg(bits)
        microscope_img = tf.image.resize(microscope_img, [config.image_height, config.image_height])
        microscope_img = _augm_geom_microscope(microscope_img,config.image_height)

        gray = tf.image.rgb_to_grayscale(microscope_img)

        mask = gray > 10
        microscope_img=_augm_color_microscope(microscope_img)

        img_bg = tf.multiply(img, tf.cast(tf.image.grayscale_to_rgb(~mask),tf.float32))
        microscope_fg = tf.multiply(tf.cast(microscope_img, dtype=tf.float32),
                              tf.cast(tf.image.grayscale_to_rgb(mask), dtype=tf.float32))

        dst = tf.add(img_bg, microscope_fg)

        return dst
    else:
        return input_img_8u
