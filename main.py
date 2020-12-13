import  os
import pandas as pd, numpy as np

import tensorflow as tf, re, math
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from model import build_model
from dataset_utils import get_dataset, count_data_items
from getscope import get_scope
from lr import get_lr_callback


print(tf.__version__)

tpu3 = "grpc://10.240.1.2:8470"
tpu2 = 'grpc://10.240.1.10:8470'

os.environ['TPU_NAME'] = tpu3

DEVICE = "TPU" #or "GPU"

# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
SEED = 0

# NUMBER OF FOLDS. USE 3, 5, OR 15
FOLDS = 4

# WHICH IMAGE SIZES TO LOAD EACH FOLD
# CHOOSE 128, 192, 256, 384, 512, 768
IMG_SIZES = [384]*FOLDS

# INCLUDE OLD COMP DATA? YES=1 NO=0
INC2019 = [0]*FOLDS
INC2018 = [0]*FOLDS

# BATCH SIZE AND EPOCHS
BATCH_SIZES = [32*2]*FOLDS
EPOCHS = [12]*FOLDS

# WHICH EFFICIENTNET B? TO USE
EFF_NETS = [6]*FOLDS

# WEIGHTS FOR FOLD MODELS WHEN PREDICTING TEST
WGTS = [1/FOLDS]*FOLDS

# TEST TIME AUGMENTATION STEPS
TTA = 11

def main():
    GCS_PATH = [None]*FOLDS; GCS_PATH2 = [None]*FOLDS
    for i,k in enumerate(IMG_SIZES):
        assert IMG_SIZES[i]==384
        GCS_PATH[i] = 'gs://kds-68d604aee5addfebbabae3fcfe5376b86711f3cd42993bd8e0dc80a5'
        GCS_PATH2[i] = 'gs://kds-4e8502fa6aa4c08b11f43ab8b42505960a29dc73fbcea54ba2bd1f9a'
    files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/train*.tfrec')))
    print(f'files_train_count={len(files_train)}')
    files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))

    REPLICAS = 8
    print(f'REPLICAS: {REPLICAS}')

    VERBOSE = 1
    DISPLAY_PLOT = True

    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_names = []
    oof_folds = []
    preds = np.zeros((int(count_data_items(files_test)), 1))

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):

        print('#' * 25)
        print('#### FOLD', fold + 1)
        print('#### Image Size %i with EfficientNet B%i and batch_size %i' %
              (IMG_SIZES[fold], EFF_NETS[fold], BATCH_SIZES[fold] * REPLICAS))

        # CREATE TRAIN AND VALIDATION SUBSETS
        files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec' % x for x in idxT])
        if INC2019[fold]:
            files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec' % x for x in idxT * 2 + 1])
            print('#### Using 2019 external data')
        if INC2018[fold]:
            files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec' % x for x in idxT * 2])
            print('#### Using 2018+2017 external data')
        np.random.shuffle(files_train)
        print('#' * 25)
        files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec' % x for x in idxV])
        files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/test*.tfrec')))

        sv = tf.keras.callbacks.ModelCheckpoint(
            'fold-%i.h5' % fold, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')
        callbacks=[sv, get_lr_callback(BATCH_SIZES[fold],replicas=REPLICAS)]
        scope = get_scope()
        with scope:
            model = build_model(dim=IMG_SIZES[fold], ef=EFF_NETS[fold])

        # TRAIN
        print('Training...')
        validation_dataset=get_dataset(files_valid, replicas=REPLICAS, augment=False, shuffle=False,
                                        repeat=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold])
        train_data_items=count_data_items(files_train)
        print(f'train_data_items={train_data_items}')
        training_dataset=get_dataset(files_train, replicas=REPLICAS, augment=True, shuffle=True, repeat=True,
                        dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold])
        history = model.fit(
            training_dataset,
            epochs=EPOCHS[fold], callbacks=callbacks,
            steps_per_epoch=train_data_items / BATCH_SIZES[fold] // REPLICAS,
            validation_data=validation_dataset,verbose=VERBOSE) #,  # class_weight = {0:1,1:2}

        print('Loading best model...')
        model.load_weights('fold-%i.h5' % fold)

        # PREDICT OOF USING TTA
        print('Predicting OOF with TTA...')
        ds_valid = get_dataset(files_valid, labeled=False, return_image_names=False, augment=True,replicas=REPLICAS,
                               repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 4)
        ct_valid = count_data_items(files_valid)
        STEPS = TTA * ct_valid / BATCH_SIZES[fold] / 4 / REPLICAS
        pred = model.predict(ds_valid, steps=STEPS, verbose=VERBOSE)[:TTA * ct_valid, ]
        oof_pred.append(np.mean(pred.reshape((ct_valid, TTA), order='F'), axis=1))
        # oof_pred.append(model.predict(get_dataset(files_valid,dim=IMG_SIZES[fold]),verbose=1))

        # GET OOF TARGETS AND NAMES
        ds_valid = get_dataset(files_valid, batch_size=BATCH_SIZES[fold], replicas=REPLICAS,
                               augment=False, repeat=False, dim=IMG_SIZES[fold],
                               labeled=True, return_image_names=True)
        oof_tar.append(np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]))
        oof_folds.append(np.ones_like(oof_tar[-1], dtype='int8') * fold)
        ds = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                         labeled=False, return_image_names=True)
        oof_names.append(np.array([img_name.numpy().decode("utf-8") for img, img_name in iter(ds.unbatch())]))

        # PREDICT TEST USING TTA
        print('Predicting Test with TTA...')
        ds_test = get_dataset(files_test, replicas=REPLICAS, labeled=False, return_image_names=False, augment=True,
                              repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 4)
        ct_test = count_data_items(files_test)
        STEPS = TTA * ct_test / BATCH_SIZES[fold] / 4 / REPLICAS
        pred = model.predict(ds_test, steps=STEPS, verbose=VERBOSE)[:TTA * ct_test, ]
        preds[:, 0] += np.mean(pred.reshape((ct_test, TTA), order='F'), axis=1) * WGTS[fold]

        # REPORT RESULTS
        auc = roc_auc_score(oof_tar[-1], oof_pred[-1])
        oof_val.append(np.max(history.history['val_auc']))
        print('#### FOLD %i OOF AUC without TTA = %.3f, with TTA = %.3f' % (fold + 1, oof_val[-1], auc))

        # PLOT TRAINING
        if DISPLAY_PLOT:
            plt.figure(figsize=(15, 5))
            plt.plot(np.arange(EPOCHS[fold]), history.history['auc'], '-o', label='Train AUC', color='#ff7f0e')
            plt.plot(np.arange(EPOCHS[fold]), history.history['val_auc'], '-o', label='Val AUC', color='#1f77b4')
            x = np.argmax(history.history['val_auc'])
            y = np.max(history.history['val_auc'])
            xdist = plt.xlim()[1] - plt.xlim()[0]
            ydist = plt.ylim()[1] - plt.ylim()[0]
            plt.scatter(x, y, s=200, color='#1f77b4')
            plt.text(x - 0.03 * xdist, y - 0.13 * ydist, 'max auc\n%.2f' % y, size=14)
            plt.ylabel('AUC', size=14)
            plt.xlabel('Epoch', size=14)
            plt.legend(loc=2)
            plt2 = plt.gca().twinx()
            plt2.plot(np.arange(EPOCHS[fold]), history.history['loss'], '-o', label='Train Loss', color='#2ca02c')
            plt2.plot(np.arange(EPOCHS[fold]), history.history['val_loss'], '-o', label='Val Loss', color='#d62728')
            x = np.argmin(history.history['val_loss'])
            y = np.min(history.history['val_loss'])
            ydist = plt.ylim()[1] - plt.ylim()[0]
            plt.scatter(x, y, s=200, color='#d62728')
            plt.text(x - 0.03 * xdist, y + 0.05 * ydist, 'min loss', size=14)
            plt.ylabel('Loss', size=14)
            plt.title('FOLD %i - Image Size %i, EfficientNet B%i, inc2019=%i, inc2018=%i' %
                      (fold + 1, IMG_SIZES[fold], EFF_NETS[fold], INC2019[fold], INC2018[fold]), size=18)
            plt.legend(loc=3)

            plt.show()




    # COMPUTE OVERALL OOF AUC

    oof = np.concatenate(oof_pred)
    true = np.concatenate(oof_tar)

    names = np.concatenate(oof_names); folds = np.concatenate(oof_folds)

    auc = roc_auc_score(true,oof)
    print('Overall OOF AUC with TTA = %.3f'%auc)

    # SAVE OOF TO DISK
    df_oof = pd.DataFrame(dict(
        image_name = names, target=true, pred = oof, fold=folds))
    df_oof.to_csv('oof.csv',index=False)
    df_oof.head()

    ds = get_dataset(files_test, replicas=REPLICAS, augment=False, repeat=False, dim=IMG_SIZES[fold],
                     labeled=False, return_image_names=True)

    image_names = np.array([img_name.numpy().decode("utf-8")
                            for img, img_name in iter(ds.unbatch())])

    submission = pd.DataFrame(dict(image_name=image_names, target=preds[:,0]))
    submission = submission.sort_values('image_name')
    submission.to_csv('submission.csv', index=False)
    submission.head()

if __name__=="__main__":
    main()