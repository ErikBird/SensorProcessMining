from tqdm import tqdm

from utils import get_patient_flow_path
import os
import pandas as pd
import csv
import random
from sklearn.utils import shuffle

path_src_train = get_patient_flow_path() / 'ABP/trainset/trainset'
path_src_dev = get_patient_flow_path() / 'ABP/testset/testset'
path_src_test = get_patient_flow_path() / 'ABP/valset/valset'
path_dest_train = 'ABP/data_train.pkl'
path_dest_dev = 'ABP/data_dev.pkl'
path_dest_test = 'ABP/data_test.pkl'
#get_patient_flow_path() /

datasets = [(path_src_train, path_dest_train), (path_src_dev, path_dest_dev), (path_src_test, path_dest_test)]
size_per_sample = 1000
seed = 42
for src_path, dest_path in tqdm(datasets):
    data = []
    with open(get_patient_flow_path() / dest_path, 'w', newline='') as dest_file:
        records = os.listdir(src_path)
        records.sort()  # inplace
        for record in records:
            with open(src_path / record, 'rb') as file:
                df = pd.read_pickle(file)
                start_index = 0
                while start_index + size_per_sample < len(df['abp'][0]):
                    # Save only every fourth sample to decrease datasize
                    should_write = random.choices([True, False], weights=[0.25, 0.75])
                    if should_write:
                        data.append(df['abp'][0][start_index:start_index + size_per_sample])
                    start_index = start_index + size_per_sample
    df = pd.DataFrame(data)
    df = shuffle(df)
    df.to_pickle(get_patient_flow_path() / dest_path)

