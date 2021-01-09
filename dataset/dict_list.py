import numpy as np

def batch_to_numpy(batch):
    batch_numpy={}
    for k, v in batch:
        batch_numpy[k]=v.numpy().astype(np.float)
    return batch_numpy

class DictList():
    def __init__(self, dict_list=None):
        self._dict={}
        if dict_list is not None:
            self._stack_first_axis(dict_list)


    def _stack_first_axis(self,dict_list):
        keys=dict_list[0].keys()

        for k in keys:
            arrays=[]
            for d in dict_list:
                arrays.append(d[k])
            self._dict[k]=np.stack(arrays,axis=0)

    @property
    def dict(self):
        return self._dict

    def extend(self, other_dict):
        for k,v in other_dict.items():
            if k in self._dict:
                self._dict[k]=np.concatenate([self._dict[k],v],axis=0)
            else:
                self._dict[k] = v

    def __getitem__(self, item):
        item_dict={}
        for k in self._dict:
                item_dict[k] = self._dict[k][item]
        return item_dict
