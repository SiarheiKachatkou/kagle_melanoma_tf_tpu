import tensorflow as tf
import unittest
from unittest.mock import patch
import getconfig
import glob
from dataset_utils import get_training_dataset
from oversample import oversample


class dict2obj(object):
    def __init__(self, d):
        self.__dict__ = d

class AugmentationTestCase(unittest.TestCase):

    @patch('getconfig.parseargs')
    @patch('augmentations.transform_geometricaly')
    @patch('dataset_utils.oversample')
    def test_augmentation(self, mock_oversample, mock_transform_geometricaly, mock_parse_args):

        img_height=128
        oversample_mult=2
        mock_parse_args.parse_args.return_value=dict2obj({'backbone':'B0','dropout_rate':0.1,
                                                          'lr_warm_up_epochs':5,'lr_max':1e-6,'lr_exp_decay':0.5,
                                                          'oversample_mult':oversample_mult,
                                                          'focal_loss_alpha':0.5,'focal_loss_gamma':4,
                                                          'image_height':img_height,'hair_prob':0.1,'microscope_prob':0.1,
                                                          'batch_size':4,'gpus':''})

        mock_oversample.side_effect=oversample
        mock_transform_geometricaly.return_value=tf.constant(value=0,shape=(img_height,img_height,3),dtype=tf.float32)
        config=getconfig.get_config()
        files=list(glob.glob(f'data/test_data/{img_height}x{img_height}/*train*'))
        train_dataset=get_training_dataset(training_fileimages=[files[0]], training_fileimages_old='', config=config)

        count=0
        batch_count=10
        for batch in train_dataset:
            if count>batch_count:
                break
            count+=1

        self.assertTrue(mock_transform_geometricaly.call_count==1)
        if oversample_mult==1:
            mock_oversample.assert_not_called()
        else:
            mock_oversample.assert_called()

    def runTest(self):
        self.test_augmentation()

if __name__=="__main__":
    unittest.TextTestRunner().run(AugmentationTestCase())