import unittest
import tensorflow as tf
from functools import partial
from unittest.mock import patch
from pathlib import Path
from config.consts import test_data_path
from fixed_save_best_callback import make_trainable, FixedSaveBestCallback, set_trainable
from model.create_model import set_backbone_trainable,create_model, BinaryFocalLoss
import efficientnet.tfkeras as efn


class SaveTest(unittest.TestCase):

    def test_save(self):
        import sys
        sys.argv=['unused', '--backbone=B0', '--dropout_rate=0.01', '--lr_max=10', '--lr_exp_decay=0.5', '--focal_loss_gamma=4', '--focal_loss_alpha=0.8', '--hair_prob=0.1', '--microscope_prob=0.01', '--lr_warm_up_epochs=5' ,'--image_height=128' ,'--work_dir=artifacts/baseline']
        from config.config import CONFIG

        filepath=str(test_data_path/"model.h5")
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)

        model = create_model(CONFIG, metrics=None, optimizer=opt, backbone_trainable=False)

        set_backbone_trainable_partial_fn=partial(set_backbone_trainable, optimizer=opt, metrics=None, cfg=CONFIG, flag=True, fine_tune_last=CONFIG.fine_tune_last)
        model=set_backbone_trainable_partial_fn(model)

        callback=FixedSaveBestCallback(filepath=filepath, set_backbone_trainable_partial_fn=set_backbone_trainable_partial_fn)
        callback.set_model(model)
        callback.on_epoch_end(epoch=1)

        set_trainable(model,True)
        #model.load_weights(filepath)
        m = tf.keras.models.load_model(filepath,custom_objects={'BinaryFocalLoss':BinaryFocalLoss}, compile=True)





if __name__=="__main__":
    unittest.main()
