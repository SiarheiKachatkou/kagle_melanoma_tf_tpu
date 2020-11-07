import tensorflow as tf


class SaveLastCallback(tf.keras.callbacks.Callback):

    def __init__(self, dir_path, fold, epochs, save_last_epochs):

        super().__init__()
        self._dir_path=dir_path
        self._fold=fold
        self._epochs=epochs
        self._save_last_epochs=save_last_epochs

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch>self._epochs-self._save_last_epochs:
            model_file_path = f'{self._dir_path}/model{self._fold}_{epoch}.h5'
            self.model.save(model_file_path)



