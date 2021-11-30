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
from utils import get_dataset_output_path, get_log_output_path, get_database_connection, get_datasequence_output_path

from collections import deque


class WindowGenerator:
    def __init__(self, process_instances, max_length, signature, exclusive_nodes: list = []):
        self.process_instances = process_instances
        self.max_length = max_length
        self.signature = signature
        self.exclusive_nodes = exclusive_nodes

    def generator(self):
        only_decisions = len(self.exclusive_nodes) > 0

        # List of the last process steps and the next step which should be predicted
        # This data structure automatically deletes steps which are not inside of the input window anymore
        history = deque(maxlen=self.max_length + 1)
        index = -1
        for instance in self.process_instances:
            index += 1
            # Fill the history with empty output on process start
            empty_step = list([tf.zeros(attr.shape[1:]) for attr in self.signature[0]])
            for _ in range(self.max_length):
                history.append(empty_step)
            step_index = -1
            for step in instance:
                step_index += 1
                if step is None:
                    print('instance %s, step %s', (index, step_index))
                    print('None input!!!')
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

                out = tuple([tuple(shaped_index), list(history)[-1][0]])

                if only_decisions:
                    potential_decision = list(history)[-2][0]
                    if any(np.array_equal(potential_decision.numpy(), node.numpy()) for node in self.exclusive_nodes):
                        yield out
                else:
                    yield out


def create_data_sequence(graph, dataset_stage: DatasetStage, number_of_instances: int):
    # Load/use the same test data also if the decisions are used
    if dataset_stage == DatasetStage.TEST_DECISION:
        dataset_stage = DatasetStage.TEST
    path = str(
        get_datasequence_output_path() / graph.__name__) + '_' + dataset_stage.value + '_' + str(
        number_of_instances)

    # with tf.device("/cpu:0"):
    try:
        with open(path + '.pkl', 'rb') as f:
            print('load data sequence')
            data_sequence = pickle.load(f)
            print('data sequence loaded')
    except:
        # To avoid memmory exhaustion if GPU to little storage
        data_sequence = list(graph.sequence_generator(process_instances=number_of_instances,
                                                      dataset_stage=dataset_stage))
        if not os.path.exists(get_datasequence_output_path()):
            os.mkdir(get_datasequence_output_path())
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data_sequence, f)
    return data_sequence, max(map(len, data_sequence))


class Dataset:
    def __init__(self, dataset, database_id: int, max_instance_length: int):
        self.data = dataset
        self.id = database_id
        self.max_instance_length = max_instance_length


class DataConvolut:

    def __init__(self, connection, graph, batch_size: int, instances_train, instances_dev, instances_test,
                 create_only_decisions: bool = False, setting=ExperimentSetting.DEFAULT,
                 anomalies_flow: float = 0.0, anomalies_sensor: float = 0.0):
        self.instances_train = instances_train
        self.instances_dev = instances_dev
        self.instances_test = instances_test
        self._anomalies_flow = anomalies_flow
        self._anomalies_flow_description = []
        self._anomalies_sensor = anomalies_sensor
        self._anomalies_sensor_description = []
        self.max_instance_length = None
        self.graph = graph
        self.signature = graph.get_signature_with_timesteps(timesteps=self.max_instance_length)
        self._persistent_signature = self.signature
        self._batch_size = batch_size
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        self._test_dataset_decisions = None
        self.connection = connection
        self._only_decisions = create_only_decisions
        self.kalman = setting in [ExperimentSetting.KALMAN_RNN, ExperimentSetting.KALMAN_CNN,
                                  ExperimentSetting.KALMAN_NORMALIZED_RNN, ExperimentSetting.KALMAN_NORMALIZED_CNN]
        self.deprecated_kalman = setting == ExperimentSetting.KALMAN
        print('Kalman %s' % self.kalman)
        self.get_datasets()

    @property
    def train(self):
        return self._train_dataset

    @property
    def dev(self):
        return self._dev_dataset

    @property
    def test(self):
        return self._test_dataset

    @property
    def test_decisions(self):
        return self._test_dataset_decisions

    @property
    def process_name(self):
        return self.graph.__name__

    @property
    def anomalies_flow(self):
        return self._anomalies_flow

    @property
    def anomalies_sensor(self):
        return self._anomalies_sensor

    @property
    def anomalies_sensor_description(self):
        return self._anomalies_sensor_description

    @property
    def anomalies_flow_description(self):
        return self._anomalies_flow_description

    @property
    def get_batch_size(self) -> int:
        return self._batch_size

    @property
    def get_max_instance_length(self) -> int:
        return self.max_instance_length

    def get_instances_by_stage(self, dataset_stage: DatasetStage):
        if dataset_stage == DatasetStage.TRAIN:
            return self.instances_train
        elif dataset_stage == DatasetStage.DEV:
            return self.instances_dev
        elif dataset_stage == DatasetStage.TEST:
            return self.instances_test
        elif dataset_stage == DatasetStage.TEST_DECISION:
            return self.instances_test
        else:
            raise NotImplementedError

    def assign_dataset_by_stage(self, dataset_stage: DatasetStage, dataset: Dataset):
        if dataset_stage == DatasetStage.TRAIN:
            self._train_dataset = dataset
        elif dataset_stage == DatasetStage.DEV:
            self._dev_dataset = dataset
        elif dataset_stage == DatasetStage.TEST:
            self._test_dataset = dataset
        elif dataset_stage == DatasetStage.TEST_DECISION:
            self._test_dataset_decisions = dataset
        else:
            raise NotImplementedError

    def get_dataset_by_stage(self, dataset_stage: DatasetStage):
        if dataset_stage == DatasetStage.TRAIN:
            return self._train_dataset
        elif dataset_stage == DatasetStage.DEV:
            return self._dev_dataset
        elif dataset_stage == DatasetStage.TEST:
            return self._test_dataset
        elif dataset_stage == DatasetStage.TEST_DECISION:
            return self._test_dataset_decisions
        else:
            raise NotImplementedError

    def get_dataset_path(self, dataset_stage: DatasetStage):
        instances = self.get_instances_by_stage(dataset_stage)
        path = str(get_dataset_output_path() / self.graph.__name__) + '_' + dataset_stage.value + '_' + str(instances)
        if self.deprecated_kalman:
            path = path + '_kalman'
        if self.kalman:
            path = path + '_extended_kalman'
        if self.anomalies_flow > 0.0:
            path = path + '_' + str(self.anomalies_flow) + 'flow'
        if self.anomalies_sensor > 0.0:
            path = path + '_' + str(self.anomalies_sensor) + 'sensor'
        return path

    def add_flow_anomalies(self, sequence):
        def rework(instance):
            anomalous_trace = instance
            if len(instance) <= 1:
                print('instance to short')
                return instance
            size = np.random.randint(low=1, high=4)
            start = np.random.randint(0, len(instance))
            dupe_sequence = anomalous_trace[start:start + size]
            anomalous_trace[start:start] = dupe_sequence
            return anomalous_trace

        def skip(instance):
            if len(instance) <= 4:
                return instance
            size = np.random.randint(1, 4)
            start = np.random.randint(0, len(instance) - size)
            end = start + size
            anomalous_trace = instance[:start] + instance[end:]
            return anomalous_trace

        def probability(instance):
            if np.random.choice([True, False], p=[self.anomalies_flow, 1.0 - self.anomalies_flow]):
                func, label = random.choice([(skip, 'Skip'), (rework, 'Rework')])
                self._anomalies_flow_description.append(label)
                return func(instance)
            else:
                self._anomalies_flow_description.append('Normal')
                return instance

        # Shift, Insert?
        # Increase WIndow Size to insert more events in rework
        self.max_instance_length = self.max_instance_length + 3
        sequence_with_anomalies = [probability(case) for case in sequence]
        return sequence_with_anomalies

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
                        if np.random.choice([True, False], p=[self.anomalies_sensor, 1.0 - self.anomalies_sensor]):
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
            self._anomalies_sensor_description.append(description)
            return output

        data = [[apply_sensor_anomalies(event) for event in instance] for instance in data]
        return data

    def visualize_flow(self, flow_description=[], sensor_description=[]):
        sequence, length = create_data_sequence(graph=self.graph, dataset_stage=DatasetStage.TRAIN,
                                                number_of_instances=self.instances_train)
        if self.anomalies_flow > 0.0:
            sequence = self.add_flow_anomalies(sequence)
        if self.anomalies_sensor > 0.0:
            sequence = self.add_sensor_anomalies(sequence)
        self.graph.visualize_sequence(sequence=sequence, flow_description=flow_description,
                                      sensor_description=sensor_description)

    def create_dataset(self, data, stage: DatasetStage):
        print('No stored dataset')
        exclusive_nodes = []
        if stage == DatasetStage.TEST_DECISION:
            # Use only the decision nodes
            exclusive_nodes = self.graph.get_decisions
        path = self.get_dataset_path(dataset_stage=stage)
        if self.anomalies_flow > 0.0 and stage == DatasetStage.TRAIN:
            print('Add anomalies on flow')
            data = self.add_flow_anomalies(data)
        if self.anomalies_sensor > 0.0 and stage == DatasetStage.TRAIN:
            print('Add anomalies on sensor')
            data = self.add_sensor_anomalies(data)
        if self.kalman:
            data = self.dataset_preprocessing(data)
        window_generator = WindowGenerator(process_instances=data,
                                           max_length=self.max_instance_length,
                                           signature=self.signature,
                                           exclusive_nodes=exclusive_nodes)

        dataset = tf.data.Dataset.from_generator(
            window_generator.generator,
            output_signature=self.signature)
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        # consumes a lot of memory:
        # dataset = dataset.cache()
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self._batch_size)  # , drop_remainder=True)
        tf.data.experimental.save(dataset, path, compression='GZIP')
        print('saved')
        number_of_event_in_sequence = sum(map(len, data))
        print('number_of_event_in_sequence: %s' % number_of_event_in_sequence)
        dataset_id = database.create_dataset(conn=self.connection,
                                             process=self.graph.__name__,
                                             num_instances=str(self.get_instances_by_stage(stage)),
                                             file_path=str(path),
                                             anomalies_flow=self.anomalies_flow,
                                             anomalies_sensor=self.anomalies_sensor,
                                             generation_setting=stage.value,
                                             max_sequence_length=self.max_instance_length,
                                             num_events=number_of_event_in_sequence)

        del dataset
        gc.collect()
        # Dataset from saved dataset is somehow faster in training
        dataset = tf.data.experimental.load(path, compression='GZIP')
        #return self.load_dataset(self.get_dataset_path(dataset_stage=stage))
        return Dataset(dataset=dataset, database_id=dataset_id, max_instance_length=self.max_instance_length)

    def get_datasets(self):
        def load_dataset(path):
            dataset = tf.data.experimental.load(path, compression='GZIP')
            #batch = next(iter(dataset))
            #print(batch[0][0].shape)
            cur = self.connection.cursor()
            full_path = str(path)
            print(full_path)
            query = 'SELECT id, max_sequence_length FROM dataset WHERE file_path="%s"' % full_path
            cur.execute(query)
            result = cur.fetchone()
            print('dataset successfully loaded')
            return Dataset(dataset=dataset, database_id=result[0], max_instance_length=result[1])

        dataset_lengths = []
        sequences = {}

        stages = [DatasetStage.TRAIN, DatasetStage.DEV, DatasetStage.TEST]
        if self._only_decisions:
            stages.append(DatasetStage.TEST_DECISION)
        for stage in stages:
            try:
                print('%s: Try loading dataset' % stage)
                dataset = load_dataset(self.get_dataset_path(dataset_stage=stage))

                self.assign_dataset_by_stage(dataset_stage=stage, dataset=dataset)
                dataset_lengths.append(self.get_dataset_by_stage(dataset_stage=stage).max_instance_length)
                print('%s: dataset loaded' % stage)
            except Exception as e:
                print(e)
                print('%s: Create Sequence' % stage)
                sequence, length = create_data_sequence(graph=self.graph, dataset_stage=stage,
                                                        number_of_instances=self.get_instances_by_stage(stage))
                dataset_lengths.append(length)
                sequences[stage] = sequence

        self.max_instance_length = max(dataset_lengths)
        print('New max instance length: %s' % self.max_instance_length)

        for stage in sequences.keys():
            print('Create dataset for stage %s' % stage)
            dataset = self.create_dataset(data=sequences[stage], stage=stage)
            self.assign_dataset_by_stage(dataset_stage=stage, dataset=dataset)

    def updata_signature(self):
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
        from multiprocessing.pool import ThreadPool as Pool
        import multiprocessing
        self.updata_signature()
        #pool = Pool(multiprocessing.cpu_count() * 100)
        #data = [[]]
        #data = map(lambda step: map(preprocess_step, step, repeat(self._persistent_signature)), data)
        data = [[preprocess_step(event, self._persistent_signature) for event in instance] for instance in data]
        # pool.starmap
        #        data = pool.starmap(lambda step: map(preprocess_step,
        #                                    zip(step, repeat(self._persistent_signature))), data)#
        return data
