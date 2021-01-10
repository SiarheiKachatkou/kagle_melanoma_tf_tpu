import os
import tensorflow as tf
from submission.submission import make_submission_dataframe, aggregate_submissions
from submission.submit import calc_auc

class CkptWithMetric():
    def __init__(self, filepath, metric, mode):
        self.filepath=filepath
        self.metric=metric
        self.mode=mode

    def __gt__(self, other):
        if self.mode=='max':
            return self.metric>other.metric
        else:
            return self.metric < other.metric


class FixedLengthList():
    def __init__(self,max_length):
        self._max_length=max_length
        self._objs=[]

    def append(self, obj):
        if len(self._objs)<self._max_length:
            self._objs.append(obj)
            return True
        else:
            if obj>min(self._objs):
                self._objs.append(obj)
                self._objs.sort(reverse=True)
                path=self._objs[-1].filepath
                if os.path.exists(path):
                    os.remove(path)
                del self._objs[-1]
                return True
            else:
                return False




class SaveBestNCallback(tf.keras.callbacks.Callback):

    def __init__(self, dir_path, fold, save_best_n, metric_name, mode, val_ttas=0, val_dataset=None):

        super().__init__()
        assert mode in ['min', 'max']
        assert save_best_n>0
        self._metric_name=metric_name
        self._mode=mode

        self._dir_path=dir_path
        self._fold=fold
        self._save_best_n=save_best_n
        self._best_ckpts=FixedLengthList(save_best_n)
        self._val_ttas=val_ttas
        self._val_dataset=val_dataset
        self._metric_history=[]

    def _get_filepath(self, epoch):
        return f'{self._dir_path}/model{self._fold}_{epoch}.h5'

    def get_filepaths(self):
        filepaths=[f.filepath for f in self._best_ckpts._objs]
        return filepaths

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if self._val_ttas==0 or self._val_ttas is None:
            if not (self._metric_name in logs):
                print('Warning: specify metrics in model.fit to save best model in SaveBestNCallback')
                return
            else:
                metric=logs[self._metric_name]
        else:
            dfs=make_submission_dataframe(self._val_dataset,self.model,self._val_ttas)
            if not isinstance(dfs,list):
                dfs=[dfs]
            metric=calc_auc(aggregate_submissions(dfs))
        print(f"\nval metric = {metric}\n")
        self._metric_history.append(metric)
        newCkpt=CkptWithMetric(self._get_filepath(epoch),metric,self._mode)

        if self._best_ckpts.append(newCkpt):
            self.model.save(newCkpt.filepath)

    def get_history(self):
        return self._metric_history



