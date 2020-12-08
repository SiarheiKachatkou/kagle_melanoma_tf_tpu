import tensorflow as tf
import unittest
from unittest.mock import patch
import getconfig
from dataset_utils import get_training_dataset


class dict2obj(object):
    def __init__(self, d):
        self.__dict__ = d

class AugmentationTestCase(unittest.TestCase):

    @patch('getconfig.parseargs')
    @patch('augmentations_geom.transform_geometricaly')
    @patch('dataset_utils.read_tfrecord')
    def test_augmentation(self, mock_transform_geometricaly, mock_parse_args, mock_read_tfrecord):

        img_height=384
        mock_parse_args.parse_args.return_value=dict2obj({'backbone':'B0','dropout_rate':0.1,
                                                          'lr_warm_up_epochs':5,'lr_max':1e-6,'lr_exp_decay':0.5,
                                                          'oversample_mult':1,
                                                          'focal_loss_alpha':0.5,'focal_loss_gamma':4,
                                                          'image_height':img_height,'hair_prob':0.1,'microscope_prob':0.1,
                                                          'batch_size':32,'gpus':''})

        image=tf.constant(value=0, shape=(img_height,img_height,3),dtype=tf.float32)
        class_label=tf.constant(0,dtype=tf.float32)
        image_name=tf.constant("test_image",dtype=tf.string)
        mock_read_tfrecord.return_value= image, class_label, image_name

        config=getconfig.get_config()

        train_dataset=get_training_dataset(training_fileimages=['no_file'], training_fileimages_old='', config=config)

        for batch in train_dataset:
            break

        mock_transform_geometricaly.assert_called()

    def runTest(self):
        self.test_augmentation()

if __name__=="__main__":
    unittest.TextTestRunner().run(AugmentationTestCase())