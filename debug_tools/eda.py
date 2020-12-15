import os
import pandas as pd
import matplotlib.pyplot as plt

data_path='data/256x256'

train_info=pd.read_csv(os.path.join(data_path,'train.csv'))

test_info=pd.read_csv(os.path.join(data_path,'test.csv'))

print(train_info.head())

print(test_info.head())


print(f'train missed values {train_info.isnull().sum()}')

print(f'test missed values {test_info.isnull().sum()}')

print('---- patient ids----')

train_info.patient_id.value_counts().plot()
plt.title('train patience id counts')
plt.show()

test_info.patient_id.value_counts().plot()
plt.title('test patience id counts')
plt.show()

test_ids=set(test_info.patient_id)
train_ids=set(train_info.patient_id)
print(f'patiens in test present in train {test_ids.intersection(train_ids)}')

print('---- age and sex----')

train_info.age_approx.hist(bins=20)
plt.title('train age')
plt.show()

test_info.age_approx.hist(bins=20)
plt.title('test age')
plt.show()

train_info['sex'].hist()
plt.title('sex in train')
plt.show()

test_info['sex'].hist()
plt.title('sex in test')
plt.show()

dbg=1