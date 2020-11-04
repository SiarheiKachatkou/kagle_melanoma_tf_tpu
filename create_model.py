import tensorflow as tf
import efficientnet.tfkeras
import tensorflow.keras.backend as K
from consts import BATCH_SIZE

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
        loss = -K.sum(self._alpha * K.pow(1. - pt_1, self._gamma) * K.log(pt_1)) \
               -K.sum((1 - self._alpha) * K.pow(pt_0, self._gamma) * K.log(1. - pt_0))
        return loss/BATCH_SIZE


def compile_model(model, metrics, cfg):
    loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)

    if cfg.l2_penalty != 0:
        regularizer = tf.keras.regularizers.l2(cfg.l2_penalty)
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

    model.compile(
        optimizer='adam',
        loss=loss_fn,  # 'categorical_crossentropy',#loss_fn
        metrics=metrics
    )

    return model


def create_model(cfg,  metrics, backbone_trainable=True):

    pretrained_model = eval(cfg.model_fn_str) 
    
    pretrained_model.trainable = backbone_trainable

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.Dropout(rate=cfg.dropout_rate),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model = compile_model(model, metrics, cfg)

    return model


def set_backbone_trainable(model, metrics, flag, cfg):
    model.layers[0].trainable = flag
    return compile_model(model, metrics, cfg)
