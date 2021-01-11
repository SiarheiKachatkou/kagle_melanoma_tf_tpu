import matplotlib.pyplot as plt
import numpy as np


def parse_model_fn(s):
    for i in range(8):
        arch='B'+str(i)
        if arch in s:
            return i
    return None


def _plot_pde1(val,x_key,y_key):
    y=np.array(val[y_key].values)
    x = np.array(val[x_key].values)
    x_unique = np.unique(x)
    y_avg = [np.mean(y[np.where(np.equal(x, x_u))]) for x_u in x_unique]
    plt.plot(x_unique, y_avg)
    plt.xlabel(x_key)
    plt.ylabel(y_key)


def _plot_pde2(val,x_key,y_key):
    y=np.array(val[y_key].values)
    x = np.array(val[list(x_key)].values)

    x_unique = np.unique(x, axis=0)
    y_avg = [np.mean(y[np.equal(np.mean(np.equal(x, x_u),axis=1),1)]) for x_u in x_unique]
    cp = plt.tricontourf(x_unique[:,0],x_unique[:,1],y_avg, 15)
    plt.colorbar(cp)
    plt.xlabel(x_key[0])
    plt.ylabel(x_key[1])

def plot_pde(val,x_key,y_key):
    if isinstance(x_key,tuple):
        _plot_pde2(val,x_key,y_key)
    else:
        _plot_pde1(val, x_key, y_key)

