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

