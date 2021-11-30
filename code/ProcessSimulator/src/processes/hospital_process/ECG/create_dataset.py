from sklearn.utils import shuffle
import pandas as pd

import os

df_train = pd.read_csv("mitbih_train.csv", header=None)
df_train = shuffle(df_train)
train_data = df_train.sample(frac=0.8)
dev_data = df_train.drop(train_data.index)
df_test = pd.read_csv("mitbih_test.csv", header=None)

normal_train = train_data[train_data[187] == 0.0]
anomal_train = train_data[train_data[187] != 0.0]
normal_dev = dev_data[dev_data[187] == 0.0]
anomal_dev = dev_data[dev_data[187] != 0.0]
normal_test = df_test[df_test[187] == 0.0]
anomal_test = df_test[df_test[187] != 0.0]

print('normal_train %s' % len(normal_train))
print('anomal_train %s' % len(anomal_train))
print('normal_dev %s' % len(normal_dev))
print('anomal_dev %s' % len(anomal_dev))
print('normal_train %s' % len(normal_test))
print('anomal_train %s' % len(anomal_test))


normal_train.to_pickle('ecg_normal_train.pkl')
anomal_train.to_pickle('ecg_anomal_train.pkl')
normal_dev.to_pickle('ecg_normal_dev.pkl')
anomal_dev.to_pickle('ecg_anomal_dev.pkl')
normal_test.to_pickle('ecg_normal_test.pkl')
anomal_test.to_pickle('ecg_anomal_test.pkl')

#df = pd.read_pickle(file_name)
