import unittest
import numpy as np
import tensorflow as tf

import efficientnet.tfkeras as efn

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
    unittest.main()

