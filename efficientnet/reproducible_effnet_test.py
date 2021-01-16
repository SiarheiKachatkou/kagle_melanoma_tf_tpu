import unittest
import numpy as np
from collections import namedtuple
import tensorflow as tf
import efficientnet.tfkeras as efn
from model.create_model import create_model

class ReproTest(unittest.TestCase):
    def test_repro(self):
        img_size = 256

        model = efn.EfficientNetB0(weights='imagenet', include_top=False)

        img = np.random.uniform(low=0.0,high=1.0,size=(1, img_size, img_size, 3)).astype(np.float32)

        output1 = model.predict(img)
        print(output1[:2,:2,:2,:2])
        output2 = model.predict(img)
        print(output2[:2,:2,:2,:2])

        self.assertAlmostEquals(output1[0,0,0,0],output2[0,0,0,0],delta=1e-3)


if __name__=="__main__":
    config = namedtuple('config', ['l2_penalty',
                                   'model_fn_str',
                                   'image_height',
                                   'use_meta',
                                    'epochs_fine_tune',
                                   'dropout_rate'
                                   ])

    img_size=256
    cfg=config(model_fn_str="efficientnet.tfkeras.EfficientNetB0(weights='imagenet', include_top=False)",
               l2_penalty=0,image_height=img_size,use_meta=0,epochs_fine_tune=0,dropout_rate=0)
    metrics=[]
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
    model=create_model(cfg, metrics, optimizer, fine_tune_last=None, backbone_trainable=True)
    img = np.random.uniform(low=0.0, high=1.0, size=(1, img_size, img_size, 3)).astype(np.float32)

    output1 = model.predict(img)
    print(output1)
    output2 = model.predict(img)
    print(output2)
    dbg=1
    #unittest.main()

