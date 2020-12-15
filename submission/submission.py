import numpy as np
import pandas as pd
import gc

import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
def return_2_values(dataset):
    def two(a1,a2,*args):
        return a1,a2
    ds=dataset.map(two,num_parallel_calls=AUTO)
    return ds

def make_submission_dataframe(test_dataset, model,repeats=1):

    '''
    ds_test = get_dataset(files_test, labeled=False, return_image_names=False, augment=True,
                          repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 4)
    ct_test = count_data_items(files_test)
    STEPS = TTA * ct_test / BATCH_SIZES[fold] / 4 / REPLICAS
    pred = model_predict(model, ds_test, steps=STEPS, verbose=VERBOSE)[:TTA * ct_test, ]
    preds[:, 0] += np.mean(pred.reshape((ct_test, TTA), order='F'), axis=1) * WGTS[fold]
    '''
    names = []
    labs = []
    for _, label, name in test_dataset.unbatch():
        names.append(name.numpy().decode('utf-8'))
        labs.append(label.numpy())

    test_dataset=test_dataset.repeat(repeats)
    preds=model.predict(return_2_values(test_dataset), verbose=True)
    preds=preds.astype(np.float)

    if preds.shape[-1]==2:
        preds=preds[:,1]
        preds=np.reshape(preds,(-1,1))

    gc.collect()

    names=np.array(names)
    labs=np.array(labs)
    names = np.reshape(names, (-1, 1))
    labs = np.reshape(labs, (-1, 1)).astype(np.int32)

    if repeats==1:
        data=np.concatenate([names,preds,labs],axis=1)
        df_submission = pd.DataFrame(data,columns=['image_name','target','labels'])
        df_submission = df_submission.sort_values(by='image_name')
        return df_submission
    else:
        df_submissions = []
        dataset_length=len(preds)//repeats
        for r in range(repeats):
            the_slice=slice(r*dataset_length,(r+1)*dataset_length)
            data = np.concatenate([names, preds[the_slice], labs], axis=1)
            df_submission = pd.DataFrame(data, columns=['image_name', 'target', 'labels'])
            df_submission = df_submission.sort_values(by='image_name')
            df_submissions.append(df_submission)
    
    return df_submissions

def avg_submissions(subms_list):
    subms=[s.sort_values(by='image_name') for s in subms_list]
    target=subms[0]['target'].values.astype(np.float)
    image_names=subms[0]['image_name']
    for i in range(1,len(subms)):
        df=subms[i]
        df_names=df['image_name']
        e=np.equal(image_names.values,df_names.values)
        assert all(e), print(f'ERROR: you are trying summing different image_names dataframe {image_names} {df_names} {e}')
        target+=df['target'].values.astype(np.float)
    
    target=target/len(subms)
    df=subms[0]
    df['target']=target
    return df
    
    
def make_submissions_all_kind(test_dataset, test_dataset_tta, models, ttas=3):
    single_model=[make_submission_dataframe(test_dataset, model) for model in models]
    
    single_model_tta=[[s] for s in single_model]
    for i in range(len(models)):
        single_model_tta[i].extend(make_submission_dataframe(test_dataset_tta, models[i],repeats=ttas))
        
    single_model_tta=[avg_submissions(s_list) for s_list in single_model_tta]
    
    all_model = avg_submissions(single_model)
    all_model_tta=avg_submissions(single_model_tta)
    return single_model[0],single_model_tta[0],all_model,all_model_tta
    
    
def calc_and_save_submissions(CONFIG,model,prefix, dataset, dataset_tta, ttas=3):
    if isinstance(model,list):
        models=model
    else:
        models=[model]
    single_model_submission,single_model_tta_submission , _,_ = make_submissions_all_kind(dataset, dataset_tta, models, ttas)

    single_model_submission.to_csv(f'{CONFIG.work_dir}/{prefix}_single_model_submission.csv')
    single_model_tta_submission.to_csv(f'{CONFIG.work_dir}/{prefix}_single_model_tta_submission.csv')