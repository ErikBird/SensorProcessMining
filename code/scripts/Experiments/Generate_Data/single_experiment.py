import os
from datetime import datetime
from enum import Enum
from itertools import repeat
import time

import pandas as pd
import tensorflow_probability as tfp
from pykalman import KalmanFilter
from keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences
import numpy as np
import tensorflow as tf
import pickle

from tqdm import tqdm

from ProcessSimulator.src.processes import HospitalSimulator, LaboratorySimulator, WindowSimulator
from ProcessSimulator.src.simulator.process_model import DatasetStage, Preprocessing, BinetTensorSpec
from database import initialize_table, create_dataset, create_experiment
from dataset import DataConvolut, create_data_sequence
from experiment import ExperimentSetting, Experiment
from external_preprocessing import preprocess_step
from utils import get_dataset_output_path, get_log_output_path, get_database_connection
from scipy.stats import chi2_contingency, pearsonr

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from collections import deque
from SensorNet.src.model import PredictionModel

if __name__ == "__main__":
    import logging

    conn = get_database_connection()
    initialize_table(conn)

    tf.get_logger().setLevel(logging.ERROR)

    graph = LaboratorySimulator()

    data_convolut = DataConvolut(connection=conn, instances_train=6000, instances_dev=1000, instances_test=2000,
                                 graph=graph,
                                 create_only_decisions=True,
                                 batch_size=128,
                                 setting=ExperimentSetting.NORMALIZED_RNN,
                                 anomalies_sensor=0.3
                                 )

    experiment = Experiment(connection=conn, graph=graph, data_convolut=data_convolut,
                            setting=ExperimentSetting.NORMALIZED_RNN, batch_size=128)
    # experiment.save_model_image()
    # experiment.train_model()
    experiment.run(only_plots=True)

