# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("mitbih_train.csv", header=None)
df_test = pd.read_csv("mitbih_test.csv", header=None)
df_train.head()

plt.plot(df_train.iloc[0, :186])
plt.plot(df_train.iloc[1, :186])
plt.plot(df_train.iloc[2, :186])
plt.plot(df_train.iloc[3, :186])
plt.plot(df_train.iloc[4, :186])
plt.plot(df_train.iloc[5, :186])
plt.plot(df_train.iloc[6, :186])
plt.plot(df_train.iloc[7, :186])
plt.plot(df_train.iloc[8, :186])
plt.plot(df_train.iloc[9, :186])
plt.show()

import seaborn as sns

sns.catplot(x=187, kind='count', data=df_train)
plt.show()
