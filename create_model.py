import tensorflow as tf
import efficientnet.tfkeras
import tensorflow.keras.backend as K

class BinaryFocalLoss():
    def __init__(self, gamma=0.2, alpha=0.25):
        self._alpha=alpha
        self._gamma=gamma
        self.__name__="BinaryFocalLoss"
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def __call__(self, y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return -K.sum(self._alpha * K.pow(1. - pt_1, self._gamma) * K.log(pt_1)) \
               -K.sum((1 - self._alpha) * K.pow(pt_0, self._gamma) * K.log(1. - pt_0))





def create_model(cfg):

    pretrained_model = eval(cfg.model_fn_str) 
    
    pretrained_model.trainable = True

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        #tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    loss_fn=BinaryFocalLoss(gamma=2.0, alpha=0.25)
    
    model.compile(
        optimizer='adam',
        loss = loss_fn,#'categorical_crossentropy',#loss_fn
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    if cfg.l2_penalty!=0:
        regularizer = tf.keras.regularizers.l2(cfg.l2_penalty)
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                  setattr(layer, attr, regularizer)
    
    return model