import tensorflow as tf
from consts import IMAGE_HEIGHT


GCS_PATH_hair_images = "data/malanoma_hairs" # "gs://kochetkov_kaggle_melanoma/malanoma_hairs"
hair_images = tf.io.gfile.glob(GCS_PATH_hair_images + '/*.png')
hair_images_tf=tf.convert_to_tensor(hair_images)

# the maximum number of hairs to augment:
n_max = 20

def hair_aug_tf(input_img, config):
    if tf.random.uniform(shape=[], maxval=1, dtype=tf.float32)<config.hair_prob:

        # Copy the input image, so it won't be changed
        img = tf.identity(input_img)

        # Unnormalize: Returning the image from 0-1 to 0-255:
        img = tf.multiply(img, 255)

        # Randomly choose the number of hairs to augment (up to n_max)
        n_hairs = tf.random.uniform(shape=[], maxval=tf.constant(n_max) + 1, dtype=tf.int32)

        im_height = tf.shape(img)[0]
        im_width = tf.shape(img)[1]

        if n_hairs == 0 or tf.random.uniform(shape=[], maxval=1, dtype=tf.float32) < 0.5:
            # Normalize the image to [0,1]
            img = tf.multiply(img, 1 / 255)
            return img

        for _ in tf.range(n_hairs):

            # Read a random hair image
            i = tf.random.uniform(shape=[], maxval=tf.shape(hair_images_tf)[0], dtype=tf.int32)
            fname = hair_images_tf[i]
            bits = tf.io.read_file(fname)
            hair = tf.image.decode_jpeg(bits)
            hair = tf.image.resize(hair, [IMAGE_HEIGHT, IMAGE_HEIGHT])
            hair = tf.image.random_flip_left_right(hair)
            hair = tf.image.random_flip_up_down(hair)

            # Random number of 90 degree rotations
            n_rot = tf.random.uniform(shape=[], maxval=4, dtype=tf.int32)
            hair = tf.image.rot90(hair, k=n_rot)

            # The hair image height and width (ignore the number of color channels)
            h_height = tf.shape(hair)[0]
            h_width = tf.shape(hair)[1]

            # The top left coord's of the region of interest (roi) where the augmentation will be performed
            roi_h0 = tf.random.uniform(shape=[], maxval=im_height - h_height + 1, dtype=tf.int32)
            roi_w0 = tf.random.uniform(shape=[], maxval=im_width - h_width + 1, dtype=tf.int32)

            # The region of interest
            roi = img[roi_h0:(roi_h0 + h_height), roi_w0:(roi_w0 + h_width)]

            # Convert the hair image to grayscale (slice to remove the trainsparency channel)
            hair2gray = tf.image.rgb_to_grayscale(hair[:, :, :3])

            # Threshold:
            mask = hair2gray > 10

            img_bg = tf.multiply(roi, tf.cast(tf.image.grayscale_to_rgb(~mask), dtype=tf.float32))
            hair_fg = tf.multiply(tf.cast(hair[:, :, :3], dtype=tf.int32),
                                  tf.cast(tf.image.grayscale_to_rgb(mask), dtype=tf.int32))

            dst = tf.add(img_bg, tf.cast(hair_fg, dtype=tf.float32))

            paddings = tf.stack(
                [[roi_h0, im_height - (roi_h0 + h_height)], [roi_w0, im_width - (roi_w0 + h_width)], [0, 0]])
            # Pad dst with zeros to make it the same shape as image.
            dst_padded = tf.pad(dst, paddings, "CONSTANT")

            # Create a boolean mask with zeros at the pixels of the augmentation segment and ones everywhere else
            mask_img = tf.pad(tf.ones_like(dst), paddings, "CONSTANT")
            mask_img = ~tf.cast(mask_img, dtype=tf.bool)

            # Make a hole in the original image at the location of the augmentation segment
            img_hole = tf.multiply(img, tf.cast(mask_img, dtype=tf.float32))

            # Inserting the augmentation segment in place of the hole
            img = tf.add(img_hole, dst_padded)

        # Normalize the image to [0,1]
        img = tf.multiply(img, 1 / 255)

        return img
    else:
        return input_img
