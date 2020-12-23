import tensorflow as tf


class SaveLastCallback(tf.keras.callbacks.Callback):

    def __init__(self, dir_path, fold, epochs, save_last_epochs):

        super().__init__()
        self._dir_path=dir_path
        self._fold=fold
        self._epochs=epochs
        self._save_last_epochs=save_last_epochs

    def _get_filepath(self, epoch):
        return f'{self._dir_path}/../trained_models/model{self._fold}_{epoch}.h5'

    def get_filepaths(self):
        filepaths=[]
        for epoch in range(self._epochs):
            if epoch>self._epochs-self._save_last_epochs:
                filepaths.append(self._get_filepath(epoch))
        return filepaths

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch>self._epochs-self._save_last_epochs:
            model_file_path = self._get_filepath(epoch)
            self.model.save(model_file_path)



