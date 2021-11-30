import csv
import itertools
import random
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import arff
from ProcessSimulator.src.simulator.process_model import ProcessSimulator, ProcessStepFunction, \
    BinetTensorSpec, \
    Preprocessing, SensorAnomalies, AttributeDataExhaustException, DatasetStage
from ProcessSimulator.src.simulator.signal_generator import perlin_noise, add_increase_by_percent
from utils import get_patient_flow_path, get_abp_train_file_path, get_abp_dev_file_path, get_abp_test_file_path, \
    get_ecg_train_normal_file_path, get_ecg_train_anormal_file_path, get_ecg_dev_normal_file_path, \
    get_ecg_dev_anormal_file_path, get_ecg_test_normal_file_path, get_ecg_test_anormal_file_path

"""

chest pain
"""


class HospitalSimulator(ProcessSimulator):
    DECIMALS = 2

    @property
    def __name__(self):
        return 'HospitalSimulator'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def heart_ecg_history(self) -> list:
            if 'ecg_correct_index' not in self.persistent_properties.keys():
                self.persistent_properties['ecg_anomaly_index'] = 0
                self.persistent_properties['ecg_correct_index'] = 0
                if self.dataset_stage == DatasetStage.TRAIN:
                    path_normal = get_ecg_train_normal_file_path()
                    path_anormal = get_ecg_train_anormal_file_path()
                elif self.dataset_stage == DatasetStage.DEV:
                    path_normal = get_ecg_dev_normal_file_path()
                    path_anormal = get_ecg_dev_anormal_file_path()
                elif self.dataset_stage == DatasetStage.TEST:
                    path_normal = get_ecg_test_normal_file_path()
                    path_anormal = get_ecg_test_anormal_file_path()
                else:
                    raise NotImplementedError
                self.persistent_properties['ecg_normal_df'] = pd.read_pickle(path_normal)
                self.persistent_properties['ecg_anormal_df'] = pd.read_pickle(path_anormal)
            anomaly = np.random.choice([True, False], p=[0.2, 0.8])
            if anomaly:
                df = self.persistent_properties['ecg_anormal_df']
                index = self.persistent_properties['ecg_anomaly_index']
                self.persistent_properties['ecg_anomaly_index'] = index + 1
            else:
                df = self.persistent_properties['ecg_normal_df']
                index = self.persistent_properties['ecg_correct_index']
                self.persistent_properties['ecg_correct_index'] = index + 1

            if index >= len(df.index):
                raise AttributeDataExhaustException

            self.process_properties['ecg_anomaly_history'] = anomaly
            output = df.iloc[index, :186].to_numpy()  # Cut Column 187 off bc. it contains the label
            return output

        def heart_abp_history(self) -> list:
            if 'abp_index' not in self.persistent_properties.keys():
                if self.dataset_stage == DatasetStage.TRAIN:
                    path = get_abp_train_file_path()
                elif self.dataset_stage == DatasetStage.DEV:
                    path = get_abp_dev_file_path()
                elif self.dataset_stage == DatasetStage.TEST:
                    path = get_abp_test_file_path()
                else:
                    raise NotImplementedError
                self.persistent_properties['abp_df'] = pd.read_pickle(path)
                self.persistent_properties['abp_index'] = 0

            N = self.persistent_properties['abp_index']
            df = self.persistent_properties['abp_df']
            self.persistent_properties['abp_index'] = N + 2
            output = df.iloc[N, :].to_numpy()
            next_measurement = df.iloc[N + 1, :].to_numpy()
            self.process_properties['heart_abp_next_measurement'] = next_measurement
            hypertension = (sum(next_measurement) / sum(output)) > 1.1  # More than 10% increase in blood pressure
            self.process_properties['heart_abp_hypertension'] = hypertension
            return output

        self.add_node(id='reception', label='Reception')
        self.add_edge(origin='start', dest='reception')

        class NewPatient(ProcessStepFunction):
            __destination__ = ['retrieve', 'create']

            def destination(self, model) -> Union[int, str]:
                if np.random.choice([True, False], p=[0.50, 0.50]):
                    return self.__destination__[0]
                else:
                    return self.__destination__[1]

        self.add_decision(origin='reception', dest=NewPatient, label='HD1: New patient?')

        self.add_node(id='retrieve', label='Retrieve Patient Information')
        self.add_edge(origin='retrieve', dest='nurse')

        self.add_attribute(node_id='retrieve', attribute_function=heart_ecg_history,
                           attributes_signature=BinetTensorSpec(shape=(186,), dtype=tf.float32,
                                                                name='sensor_ecg_history'))
        self.add_attribute(node_id='retrieve', attribute_function=heart_abp_history,
                           attributes_signature=BinetTensorSpec(shape=(1000,), dtype=tf.float32,
                                                                name='sensor_abp_history'))

        class NurseJudgement(ProcessStepFunction):
            __destination__ = ['wait', 'triage']

            def destination(self, model) -> Union[int, str]:
                if 'ecg_anomaly_history' in model.process_properties.keys() and model.process_properties[
                    'ecg_anomaly_history']:
                    return self.__destination__[1]
                else:
                    return self.__destination__[0]

        self.add_node(id='create', label='Create new Patient File')
        self.add_edge(origin='create', dest='nurse')

        self.add_node(id='nurse', label='Nurse')
        self.add_decision(origin='nurse', dest=NurseJudgement, label='HD2: Prestressed patient?')

        def patient_age(self) -> list:
            old = np.random.choice([False, True], p=[0.1, 0.9])
            if old:
                age = np.random.normal(80, 20)
            else:
                age = np.random.normal(35, 20)
            self.process_properties['patient_age'] = age
            return [age]

        self.add_attribute(node_id=['retrieve', 'create'], attribute_function=patient_age,
                           attributes_signature=BinetTensorSpec(shape=(1,), dtype=tf.float32, name='patient_age'))

        self.add_node(id='wait', label='Waiting')

        self.add_node(id='triage', label='Triage')
        self.add_edge(origin='wait', dest='triage')

        def heart_ecg(self) -> list:
            if 'ecg_correct_index' not in self.persistent_properties.keys():
                self.persistent_properties['ecg_anomaly_index'] = 0
                self.persistent_properties['ecg_correct_index'] = 0
                if self.dataset_stage == DatasetStage.TRAIN:
                    path_normal = get_ecg_train_normal_file_path()
                    path_anormal = get_ecg_train_anormal_file_path()
                elif self.dataset_stage == DatasetStage.DEV:
                    path_normal = get_ecg_dev_normal_file_path()
                    path_anormal = get_ecg_dev_anormal_file_path()
                elif self.dataset_stage == DatasetStage.TEST:
                    path_normal = get_ecg_test_normal_file_path()
                    path_anormal = get_ecg_test_anormal_file_path()
                else:
                    raise NotImplementedError
                self.persistent_properties['ecg_normal_df'] = pd.read_pickle(path_normal)
                self.persistent_properties['ecg_anormal_df'] = pd.read_pickle(path_anormal)
            anomaly = np.random.choice([True, False], p=[0.2, 0.8])
            if anomaly:
                df = self.persistent_properties['ecg_anormal_df']
                index = self.persistent_properties['ecg_anomaly_index']
                self.persistent_properties['ecg_anomaly_index'] = index + 1
            else:
                df = self.persistent_properties['ecg_normal_df']
                index = self.persistent_properties['ecg_correct_index']
                self.persistent_properties['ecg_correct_index'] = index + 1

            if index >= len(df.index):
                raise AttributeDataExhaustException

            self.process_properties['ecg_anomaly'] = anomaly
            output = df.iloc[index, :186].to_numpy()  # Cut Column 187 off bc. it contains the label
            return output

        def heart_abp(self) -> list:
            if self.dataset_stage == DatasetStage.TRAIN:
                path = get_abp_train_file_path()
            elif self.dataset_stage == DatasetStage.DEV:
                path = get_abp_dev_file_path()
            elif self.dataset_stage == DatasetStage.TEST:
                path = get_abp_test_file_path()
            else:
                raise NotImplementedError

            if 'heart_abp_next_measurement' in self.process_properties.keys():
                return self.process_properties['heart_abp_next_measurement']
            else:
                if 'abp_index' not in self.persistent_properties.keys():
                    self.persistent_properties['abp_index'] = 0
                    self.persistent_properties['abp_df'] = pd.read_pickle(path)
                with open(path, 'r') as f:
                    N = self.persistent_properties['abp_index']
                    df = self.persistent_properties['abp_df']
                    self.persistent_properties['abp_index'] = N + 1
                    output = df.iloc[N, :].to_numpy()
                    return output

        self.add_attribute(node_id='triage', attribute_function=heart_ecg,
                           attributes_signature=BinetTensorSpec(shape=(186,), dtype=tf.float32,
                                                                name='sensor_ecg'))
        self.add_attribute(node_id='triage', attribute_function=heart_abp,
                           attributes_signature=BinetTensorSpec(shape=(1000,), dtype=tf.float32,
                                                                name='sensor_abp'))

        self.add_node(id='diagnosis', label='Diagnosis')
        self.add_edge(origin='triage', dest='diagnosis')

        class Diagnosis(ProcessStepFunction):
            __destination__ = ['prescribe', 'discharge']

            def destination(self, model) -> Union[int, str]:
                if model.process_properties['ecg_anomaly'] or \
                        ('heart_abp_hypertension' in model.process_properties.keys()
                         and model.process_properties['heart_abp_hypertension']):
                    return self.__destination__[0]
                else:
                    return self.__destination__[1]

        self.add_decision(origin='diagnosis', dest=Diagnosis, label='HD3: Physical problems found?')

        self.add_node(id='prescribe', label='Prescribe Therapie')

        class Prescribe(ProcessStepFunction):
            __destination__ = ['clinic', 'hypertension', 'emergency']

            def destination(self, model) -> Union[int, str]:
                if model.process_properties['ecg_anomaly']:
                    if model.process_properties['patient_age'] < 50:
                        return self.__destination__[2]
                    return self.__destination__[0]
                else:
                    return self.__destination__[1]

        self.add_decision(origin='prescribe', dest=Prescribe, label='HD4: How to handle the patient?')

        self.add_node(id='hypertension', label='Prescribe Hypertension')
        self.add_edge(origin='hypertension', dest='discharge')

        self.add_node(id='clinic', label='Clinic')
        self.add_edge(origin='clinic', dest='end')
        self.add_node(id='discharge', label='Discharge Patient')
        self.add_edge(origin='discharge', dest='end')
        self.add_node(id='emergency', label='Emergency Department')
        self.add_edge(origin='emergency', dest='end')


        self.add_node(id='end', label='End')

        class Finish(ProcessStepFunction):
            __destination__ = ['start']

            def destination(self, model) -> Union[int, str]:
                model.reset()
                return model.last_visited_step_id

        self.add_decision(origin='end', dest=Finish)
