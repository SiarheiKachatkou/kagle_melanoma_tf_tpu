import tensorflow as tf

def make_trainable(method_fn):
    def decorated_method_fn(*args,**kwargs):
        self_value=args[0]
        self_value.model.trainable=True
        method_fn(*args,**kwargs)
        self_value.model=self_value._set_backbone_trainable_partial_fn(self_value.model)
    return decorated_method_fn



class FixedSaveBestCallback(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self,set_backbone_trainable_partial_fn, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._set_backbone_trainable_partial_fn=set_backbone_trainable_partial_fn

    @make_trainable
    def on_epoch_end(self, *args, **kwargs):
        super().on_epoch_end(*args, **kwargs)



