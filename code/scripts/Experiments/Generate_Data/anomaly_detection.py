from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score, \
    roc_auc_score
import pandas as pd
from matplotlib import pyplot as plt

from ProcessSimulator.src.processes import LaboratorySimulator
from ProcessSimulator.src.processes.hospital_process.hospital_process import HospitalSimulator
from ProcessSimulator.src.simulator.process_model import BinetTensorSpec
import numpy as np
import gc
import os
from itertools import repeat
import random

import numpy as np
import tensorflow as tf
import pickle

import database
from ProcessSimulator.src.simulator.process_model import DatasetStage, Preprocessing, BinetTensorSpec
from ProcessSimulator.src.simulator.signal_generator import add_stuck_at_constant, add_trend, add_spike, \
    add_spike_cluster, add_stuck_at_zero
from experiment import ExperimentSetting
from external_preprocessing import preprocess_step
from utils import get_dataset_output_path, get_log_output_path, get_database_connection, get_datasequence_output_path, \
    get_model_output_path, get_assets_path


class WindowGenerator:
    def __init__(self, process_instances, max_length, signature, flow_anomaly_probability, only_first_anomaly=False):
        # self.process_instances = process_instances
        self.max_length = max_length
        self.signature = signature
        self.anomalies_flow = flow_anomaly_probability
        self.process_instances = self.add_flow_anomalies(process_instances)
        self.only_first_anomaly = only_first_anomaly

    def add_flow_anomalies(self, sequence):
        def rework(instance):
            anomalous_trace = instance
            anomaly_status = [False] * len(anomalous_trace)
            size = np.random.randint(low=1, high=4)
            start = np.random.randint(0, len(instance))
            dupe_sequence = anomalous_trace[start:start + size]
            anomalous_trace[start:start] = dupe_sequence
            anomaly_status[start:start] = [True] * len(dupe_sequence)
            return [anomaly_status, anomalous_trace]

        def skip(instance):
            size = np.random.randint(1, 4)
            start = np.random.randint(0, len(instance) - size)
            end = start + size
            anomalous_trace = instance[:start] + instance[end:]
            anomaly_status = [False] * len(anomalous_trace)
            anomaly_status[start] = True
            return [anomaly_status, anomalous_trace]

        def probability(instance):
            if np.random.choice([True, False], p=[self.anomalies_flow, 1.0 - self.anomalies_flow]):
                func, label = random.choice([(skip, 'Skip'), (rework, 'Rework')])
                return func(instance)
            else:
                return [[False] * len(instance), instance]

        # Shift, Insert?
        # Increase WIndow Size to insert more events in rework
        sequence_with_anomalies = [probability(case) for case in sequence]
        return sequence_with_anomalies

    def generator(self):

        # List of the last process steps and the next step which should be predicted
        # This data structure automatically deletes steps which are not inside of the input window anymore
        history = deque(maxlen=self.max_length + 1)
        index = -1
        for anomaly_labels, instance in self.process_instances:
            index += 1
            # Fill the history with empty output on process start
            empty_step = list([tf.zeros(attr.shape[1:]) for attr in self.signature[0]])
            for _ in range(self.max_length):
                history.append(empty_step)
            step_index = -1
            for anomaly_label, step in zip(anomaly_labels, instance):
                step_index += 1
                history.append(step)

                input_labels = []
                for input in list(history)[:-1]:
                    input_labels.append(input)

                shaped_index = []
                for index in range(len(input_labels[0])):
                    attribute = []
                    for event in input_labels:
                        attribute.append(event[index])
                    shaped_index.append(tf.stack(attribute))

                out = tuple([tuple(shaped_index), tuple([list(history)[-1][0], anomaly_label])])
                yield out
                if self.only_first_anomaly and anomaly_label:
                    break


class anomaly_experiment:
    def __init__(self, connection, graph, batch_size, instances, preprocessing, train_flow_anomalies=0.0,
                 train_sensor_anomalies=0.0, eval_sensor_anomalies=0.3, eval_flow_anomalies=0.3, kalman=False,
                 only_first_anomaly=False):
        self.instances = instances
        self.preprocessing = preprocessing
        self.train_sensor_anomalies = train_sensor_anomalies
        self.train_flow_anomalies = train_flow_anomalies
        self.eval_sensor_anomalies = eval_sensor_anomalies
        self.eval_flow_anomalies = eval_flow_anomalies
        self.kalman = kalman
        self.graph = graph
        path = get_model_output_path() / self.experiment_name
        self.model = tf.keras.models.load_model(path)
        self.trained_timesteps = self.model.layers[0].input_shape[0][1]
        self.signature = graph.get_signature_with_timesteps(timesteps=self.trained_timesteps)
        self._persistent_signature = self.signature
        self.batch_size = batch_size
        self.connection = connection


    def update_signature(self):
        signature = list(self._persistent_signature[0])
        offset = 0
        print(signature)
        for index, spec in enumerate(self._persistent_signature[0]):
            if 'sensor' in spec.name:
                signature[index + offset:index + offset] = [BinetTensorSpec(shape=(spec.shape[0], 1), dtype=spec.dtype,
                                                                            name=spec.name + '_kalman',
                                                                            preprocessing=Preprocessing.DEFAULT,
                                                                            sensor_anomalies=spec.sensor_anomalies)]
                offset += 1
                print(signature)
        self.signature = (tuple(signature), self.signature[1])

    def dataset_preprocessing(self, data):
        self.update_signature()
        data = [[preprocess_step(event, self._persistent_signature) for event in instance] for instance in data]
        return data

    def add_sensor_anomalies(self, data):
        def apply_sensor_anomalies(input):
            output = []
            description = []
            for attribute, spec in zip(input, self.signature[0]):
                if 'sensor' in spec.name:
                    if tf.is_tensor(attribute):
                        attribute = attribute.numpy()
                    if not isinstance(attribute, np.ndarray):
                        attribute = np.array(attribute)
                    if attribute.nonzero()[0].size != 0:
                        if np.random.choice([True, False],
                                            p=[self.eval_sensor_anomalies, 1.0 - self.eval_sensor_anomalies]):
                            func, label = random.choice([(add_stuck_at_constant, 'stuck_at_constant'),
                                                         (add_trend, 'trend'),
                                                         (add_spike, 'spike'),
                                                         (add_spike_cluster, 'spike_cluster'),
                                                         (add_stuck_at_zero, 'stuck_at_zero')])
                            description.append(label)
                            value = func(attribute)
                            output.append(value)
                        else:
                            description.append('normal')
                            output.append(attribute)
                    else:
                        output.append(attribute)
                else:
                    output.append(attribute)
            return output

        data = [[apply_sensor_anomalies(event) for event in instance] for instance in data]
        return data

    def create_data_sequence(self, dataset_stage: DatasetStage, number_of_instances: int):
        # Load/use the same test data also if the decisions are used
        if dataset_stage == DatasetStage.TEST_DECISION:
            dataset_stage = DatasetStage.TEST
        path = str(
            get_datasequence_output_path() / self.graph.__name__) + '_' + dataset_stage.value + '_' + str(
            number_of_instances)

        # with tf.device("/cpu:0"):
        try:
            with open(path + '.pkl', 'rb') as f:
                print('load data sequence')
                data_sequence = pickle.load(f)
                print('data sequence loaded')
        except:
            # To avoid memmory exhaustion if GPU to little storage
            data_sequence = list(self.graph.sequence_generator(process_instances=number_of_instances,
                                                               dataset_stage=dataset_stage))
            if not os.path.exists(get_datasequence_output_path()):
                os.mkdir(get_datasequence_output_path())
            with open(path + '.pkl', 'wb') as f:
                pickle.dump(data_sequence, f)
        return data_sequence

    @property
    def experiment_name(self):
        name = str(self.instances) \
               + '_' + self.preprocessing + '_' + self.graph.__name__
        if self.kalman:
            name = name + '_extended_kalman'
        if self.train_flow_anomalies > 0.0:
            name = name + '_' + str(self.train_flow_anomalies) + 'flow'
        if self.train_sensor_anomalies > 0.0:
            name = name + '_' + str(self.train_sensor_anomalies) + 'sensor'
        return name

    def run(self):



        data_sequence = self.create_data_sequence(dataset_stage=DatasetStage.TEST,
                                                  number_of_instances=2000)
        data_sequence = self.add_sensor_anomalies(data_sequence)
        if self.preprocessing in [ExperimentSetting.KALMAN_NORMALIZED_RNN.value, ExperimentSetting.KALMAN_NORMALIZED_CNN.value]:
            data_sequence = self.dataset_preprocessing(data_sequence)

        data_generator = WindowGenerator(process_instances=data_sequence, max_length=self.trained_timesteps,
                                         signature=self.signature, flow_anomaly_probability=self.eval_flow_anomalies,
                                         only_first_anomaly=only_first_anomaly)

        configured_signature = (
            self.signature[0], (self.signature[1], BinetTensorSpec(shape=(), dtype=tf.bool, name='anomaly')))
        dataset_anomalies = tf.data.Dataset.from_generator(
            data_generator.generator,
            output_signature=configured_signature)
        dataset_anomalies = dataset_anomalies.batch(128, drop_remainder=True)
        predicted_anomalies = []

        anomaly_labels = []
        mse = []
        for input, target in dataset_anomalies:
            predictions = self.model.predict(input)
            for prediction_index in range(predictions.shape[0]):
                # label_id = np.argmax(target[0][prediction_index])
                score = np.square(np.subtract(predictions[prediction_index], target[0][prediction_index])).mean()
                # score = max(predictions[prediction_index]) - predictions[prediction_index][label_id]
                mse.append(score)
                anomaly_labels.append(target[1][prediction_index].numpy())

        avg_mse = sum(mse) / len(mse)
        print(avg_mse)
        predicted_anomalies = [error > avg_mse for error in mse]
        fig, ax = plt.subplots()
        for index, error, label in zip(range(len(mse)), mse, anomaly_labels):
            if label:
                ax.plot(index, error, marker='.', color='r')
            else:
                ax.plot(index, error, marker='.', color='g')
        ax.hlines(avg_mse, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100)  # , label='Threshold'
        # ax.legend()
        plt.title("Prediction error for normal and anomalous events")
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Data point index")
        print(f'F1: {f1_score(anomaly_labels, predicted_anomalies):.2f}')
        if only_first_anomaly:
            label_anomalies = 'only_first'
        else:
            label_anomalies = 'all'
        name = str(self.instances) + self.graph.__name__ + '_' + self.preprocessing + '_' + label_anomalies+'.pdf'
        fig.savefig(get_assets_path() / 'Scatterplots' / 'AnomalyDetection' / name, format='pdf')
        plt.show()
        '''
        database.create_anomaly_experiment(conn=self.connection,
                                           preprocessing=self.preprocessing,
                                           train_sensor_anomalies=self.train_sensor_anomalies,
                                           train_flow_anomalies=self.train_flow_anomalies,
                                           eval_sensor_anomalies=self.eval_sensor_anomalies,
                                           eval_flow_anomalies=self.eval_flow_anomalies,
                                           anomalies=label_anomalies,
                                           graph=str(self.instances) + self.graph.__name__,
                                           accuracy=accuracy_score(anomaly_labels, predicted_anomalies),
                                           kappa=cohen_kappa_score(anomaly_labels, predicted_anomalies),
                                           precision=precision_score(anomaly_labels, predicted_anomalies),
                                           recall=recall_score(anomaly_labels, predicted_anomalies),
                                           f1=f1_score(anomaly_labels, predicted_anomalies))'''


conn = get_database_connection()

for instances in [10000, 6000]:
    for only_first_anomaly in [True, False]:  #
        for sensor_anomalies, flow_anomalies in [(0.3, 0.3)]:
            for graph in [
                LaboratorySimulator,
                HospitalSimulator
            ]:
                graph = graph()
                for setting in [
                    # ExperimentSetting.CONVOLUTION,
                    # ExperimentSetting.DEEP_CNN,
                    # ExperimentSetting.RNN,
                    # ExperimentSetting.DEFAULT,
                    # ExperimentSetting.NORMALIZE,
                    ExperimentSetting.BASELINE_SENSOR,
                    ExperimentSetting.BASELINE_ID,
                    ExperimentSetting.BASELINE_ATTRIBUTE,
                    # ExperimentSetting.DEEP_CNN_NO_POOL,
                    # ExperimentSetting.DEEP_FFN,
                    ExperimentSetting.NORMALIZED_RNN,
                    ExperimentSetting.NORMALIZED_CNN,
                    # ExperimentSetting.KALMAN,
                    #ExperimentSetting.KALMAN_RNN,
                    #ExperimentSetting.KALMAN_CNN,
                    #ExperimentSetting.KALMAN_NORMALIZED_RNN,
                    #ExperimentSetting.KALMAN_NORMALIZED_CNN,

                ]:
                    database.initialize_table(conn)
                    experiment = anomaly_experiment(connection=conn, graph=graph, batch_size=128, instances=instances,
                                                    preprocessing=setting.value, train_flow_anomalies=flow_anomalies,
                                                    train_sensor_anomalies=sensor_anomalies, kalman=False,
                                                    only_first_anomaly=only_first_anomaly)
                    print('ExperimentSetting: %s' % setting.value)
                    experiment.run()  # only_plots=True
                    del experiment
                    gc.collect()
