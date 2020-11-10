import numpy as np
import pandas as pd
from tqdm import tqdm

def make_submission_dataframe(test_dataset, model):

    preds=[]
    names=[]
    labs=[]
    for batch in tqdm(test_dataset):
        images, labels, image_names = batch
        labs.extend(labels.numpy())
        image_names = image_names.numpy()

        predictions = model.predict(images, batch_size=64, workers=8, use_multiprocessing=True)
        preds.extend(predictions)
        names.extend(image_names)

    names=[n.decode('utf-8') for n in names]
    names=np.array(names)
    preds=np.array(preds)
    labs=np.array(labs)
    #labs=np.argmax(labs,axis=1)
    names = np.reshape(names, (-1, 1))
    labs = np.reshape(labs, (-1, 1))
    data=np.concatenate([names,preds,labs],axis=1)
    df_submission = pd.DataFrame(data,columns=['image_name','target','labels'])
    df_submission = df_submission.sort_values(by='image_name')
    
    return df_submission

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
        for t in range(ttas):
            single_model_tta[i].append(make_submission_dataframe(test_dataset_tta, models[i]))
        
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