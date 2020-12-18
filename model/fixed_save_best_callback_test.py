import unittest
from unittest.mock import patch
from pathlib import Path
from config.consts import test_data_path
from fixed_save_best_callback import make_trainable, FixedSaveBestCallback
import efficientnet.tfkeras as efn


class SaveTest(unittest.TestCase):

    def test_save(self):
        filepath=str(test_data_path/"model.h5")
        model = efn.EfficientNetB0(weights='imagenet', include_top=False)

        def set_backbone_trainable_partial_fn(model):
            model.trainable=True
            model.layers[1].trainable=False
            return model
        model=set_backbone_trainable_partial_fn(model)

        callback=FixedSaveBestCallback(filepath=filepath, set_backbone_trainable_partial_fn=set_backbone_trainable_partial_fn)
        callback.set_model(model)
        callback.on_epoch_end(epoch=1)

        self.assertTrue(model.trainable)
        self.assertTrue(not model.layers[1].trainable)

        model.load_weights(filepath)



if __name__=="__main__":
    unittest.main()
