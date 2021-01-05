import tensorflow as tf
import efficientnet.tfkeras
import efficientnet.tfkeras as efn
import os
import tempfile
import tensorflow.keras.backend as K
import multiprocessing as mp
from config.consts import use_amp
from model.sparceauc import SparceAUC


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
    tmp_weights_path = os.path.join(tempfile.gettempdir(), f'tmp_weights{mp.current_process().pid}.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

class BinaryFocalLoss():
    def __init__(self, batch_size, gamma=0.2, alpha=0.25):
        # alpha - account for imbalance, alpha=[0,1] how much positive more important then negative"
        # gamma - focus on hard examples increasing gamma increase focus on hard examples"
        self._alpha=alpha
        self._gamma=gamma
        self._batch_size=batch_size
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
        return loss/self._batch_size


def compile_model(model, metrics, cfg, optimizer):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()#BinaryCrossentropy()#BinaryFocalLoss(gamma=cfg.focal_loss_gamma,alpha=cfg.focal_loss_alpha, batch_size=cfg.batch_size)#label_smoothing=0.05)


    if use_amp:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6]


def create_model(cfg,  metrics, optimizer, fine_tune_last=None, backbone_trainable=True):

    pretrained_model = eval(cfg.model_fn_str) 
    if cfg.l2_penalty != 0:
        regularizer = tf.keras.regularizers.l2(cfg.l2_penalty)
        pretrained_model=add_regularization(pretrained_model, regularizer)

    pretrained_model.trainable = False

    image=tf.keras.Input(shape=(cfg.image_height,cfg.image_height,3),name='image')
    sex=tf.keras.Input(shape=(2,),name='sex')
    age = tf.keras.Input(shape=(1,), name='age')
    anatom=tf.keras.Input(shape=(10,), name='anatom_site')
    meta_feature=tf.concat([sex,age,anatom],axis=1)

    training=cfg.epochs_fine_tune==0
    features=pretrained_model(image,training=training)

    head=tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
    ])

    image_model_output = head(features)

    meta_hidden1 = 64
    meta_hidden2 = 32

    meta_feature=tf.keras.layers.Dense(meta_hidden1, activation='relu')(meta_feature)
    meta_feature=tf.keras.layers.BatchNormalization()(meta_feature)
    meta_feature = tf.keras.layers.Dense(meta_hidden2, activation='relu')(meta_feature)
    features=tf.concat([meta_feature,image_model_output],axis=1)
    #features = image_model_output

    features = tf.keras.layers.Dropout(rate=cfg.dropout_rate)(features)
    output = tf.keras.layers.Dense(2, activation='softmax')(features)

    model=tf.keras.Model(inputs=[image,sex, age,anatom],outputs=output)

    if cfg.l2_penalty != 0:
        regularizer = tf.keras.regularizers.l2(cfg.l2_penalty)
        model=add_regularization(model, regularizer)

    return set_backbone_trainable(model, metrics, optimizer, backbone_trainable, cfg, fine_tune_last=fine_tune_last)


def set_backbone_trainable(model, metrics, optimizer, flag, cfg, fine_tune_last=None):
    if flag:
        for l in model.layers:
            if hasattr(l,'trainable'):
                l.trainable=flag
    return compile_model(model, metrics, cfg, optimizer)


def load_model(filepath):
    m = tf.keras.models.load_model(filepath, custom_objects={'BinaryFocalLoss': BinaryFocalLoss, 'SparceAUC':SparceAUC}, compile=True)
    m.trainable = False
    return m