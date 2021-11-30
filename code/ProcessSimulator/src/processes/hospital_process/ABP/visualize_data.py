from utils import get_patient_flow_path
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt

path_train = get_patient_flow_path() / 'ABP/trainset/trainset'
size_per_sample = 1000

train_records = os.listdir(path_train)
train_records.sort()  # inplace


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

N = 50
cmap = get_cmap(N)
averages_x = []
averages_y = []
with open(get_patient_flow_path() / 'ABP/sensor_train.csv', 'w', newline='') as myfile:
    for n, record in enumerate(train_records[:N]):
        with open(path_train / record, 'rb') as file:
            df = pd.read_pickle(file)
            start_index = 0
            average_x = []
            average_y = []
            while start_index + size_per_sample < len(df['abp'][0]) / 4:
                wr = csv.writer(myfile)
                data = df['abp'][0][start_index:start_index + size_per_sample]
                average_x.append(sum(data) / size_per_sample)
                average_y.append(n)
                plt.plot(range(len(data)), data, color=cmap(n), alpha=0.7)
                start_index = start_index + size_per_sample
            averages_x.append(average_x)
            averages_y.append(average_y)
# label = "line 1")
plt.show()
for n, average in enumerate(averages_x):
    plt.scatter(average, averages_y[n], alpha=0.1, color=cmap(n))
plt.show()
