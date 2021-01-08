import tensorflow as tf

class SparceAUC(tf.keras.metrics.AUC):

    def __init__(self, **kwargs):
        super(SparceAUC, self).__init__(**kwargs)


    def update_state(self, y_true, y_pred, sample_weight=None):

        super().update_state(y_true, y_pred[:,1], sample_weight)

    def result(self):
        return super().result()
