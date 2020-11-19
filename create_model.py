import tensorflow as tf
import efficientnet.tfkeras
import efficientnet.tfkeras as efn
import os
import tempfile
import tensorflow.keras.backend as K
from consts import BATCH_SIZE


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

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


def compile_model(model, metrics, cfg, lr=None):
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

    learning_rate = cfg.lr_start if lr is None else lr

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )

    return model

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6]


def create_model(cfg,  metrics, backbone_trainable=True, lr=None):

    pretrained_model = eval(cfg.model_fn_str) 
    if cfg.l2_penalty != 0:
        regularizer = tf.keras.regularizers.l2(cfg.l2_penalty)
        pretrained_model=add_regularization(pretrained_model, regularizer)

    pretrained_model.trainable = backbone_trainable

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.Dropout(rate=cfg.dropout_rate),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    if cfg.l2_penalty != 0:
        regularizer = tf.keras.regularizers.l2(cfg.l2_penalty)
        model=add_regularization(model, regularizer)

    model = compile_model(model, metrics, cfg, lr)

    return model


def set_backbone_trainable(model, metrics, flag, cfg):
    model.layers[0].trainable = flag
    return compile_model(model, metrics, cfg)
