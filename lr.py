import math
import tensorflow as tf

def get_lrfn(CONFIG):
    def lrfn(epoch):

        LR_SUSTAIN_EPOCHS = 0

        if epoch < CONFIG.lr_warm_up_epochs:
            lr = (CONFIG.lr_max - CONFIG.lr_start) / CONFIG.lr_warm_up_epochs * epoch + CONFIG.lr_start
        elif epoch < CONFIG.lr_warm_up_epochs + LR_SUSTAIN_EPOCHS:
            lr = CONFIG.lr_max
        else:
            lr = (CONFIG.lr_max - CONFIG.lr_min) * CONFIG.lr_exp_decay**(epoch - CONFIG.lr_warm_up_epochs - LR_SUSTAIN_EPOCHS) + CONFIG.lr_min
        return lr
    return lrfn


def get_cycling_lrfn(CONFIG):

    def lrfn(epoch):

        cycle = math.floor(1+epoch/(2*CONFIG.stepsize))
        x = math.fabs(epoch/CONFIG.stepsize - 2*cycle + 1)
        lr = CONFIG.lr_min + (CONFIG.lr_max-CONFIG.lr_min)*max([0,1-x])*CONFIG.lr_exp_decay**epoch

        return lr
    return lrfn

