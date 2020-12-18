import unittest
import tensorflow as tf
from functools import partial
from unittest.mock import patch
from pathlib import Path
from config.consts import test_data_path
from model.create_model import set_backbone_trainable,create_model, load_model
from model.sparceauc import SparceAUC
import efficientnet.tfkeras as efn
from tensorflow.keras.callbacks import ModelCheckpoint

class SaveTest(unittest.TestCase):

    def test_save(self):
        import sys
        sys.argv=['unused', '--backbone=B0', '--dropout_rate=0.01', '--lr_max=10', '--lr_exp_decay=0.5', '--focal_loss_gamma=4', '--focal_loss_alpha=0.8', '--hair_prob=0.1', '--microscope_prob=0.01', '--lr_warm_up_epochs=5' ,'--image_height=128' ,'--work_dir=artifacts/baseline']
        from config.config import CONFIG

        filepath=str(test_data_path/"model.h5")
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)

        metrics=[SparceAUC(name="auc")]
        model = create_model(CONFIG, metrics=metrics, optimizer=opt, backbone_trainable=False)

        model=set_backbone_trainable(model, optimizer=opt, metrics=metrics, cfg=CONFIG, flag=True, fine_tune_last=CONFIG.fine_tune_last)

        callback=ModelCheckpoint(filepath=filepath)
        callback.set_model(model)
        callback.on_epoch_end(epoch=1)

        m = load_model(filepath)

        self.assertTrue(m is not None)





if __name__=="__main__":
    unittest.main()
