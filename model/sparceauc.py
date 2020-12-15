import tensorflow as tf

class SparceAUC(tf.keras.metrics.AUC):

    def __init__(self,
                 **kwargs):
        super(SparceAUC, self).__init__(name="auc", num_thresholds=2000)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_one_hot = tf.compat.v1.one_hot(tf.cast(y_true, dtype=tf.int32), depth=2)
        y_true_one_hot = tf.keras.backend.squeeze(y_true_one_hot, axis=1)

        super().update_state(y_true_one_hot, y_pred, sample_weight)

    def result(self):
        return super().result()
