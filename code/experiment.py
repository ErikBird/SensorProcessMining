import os
from datetime import datetime
from enum import Enum
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, f1_score, \
    roc_auc_score
import dot2tex
import tensorflow as tf
from ProcessSimulator.src.simulator.process_model import Preprocessing, BinetTensorSpec
from SensorNet.src.model import PredictionModel
from SensorNet.src.util import PlotTraining, lr_decay
from database import create_experiment
from utils import get_log_output_path, get_assets_path, get_model_output_path, get_matplotlib_size
import numpy as np
from collections import Counter
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import defaultdict


class ExperimentSetting(Enum):
    DEFAULT = 'default'
    BASELINE_ID = 'baseline_id'
    BASELINE_ATTRIBUTE = 'baseline_attr'
    BASELINE_SENSOR = 'baseline_sensor'
    CONVOLUTION = 'convolution'
    RNN = 'rnn'
    KALMAN = 'kalman'
    NORMALIZE = 'normalize'
    HAMPEL = 'hampel'
    DEEP_CNN = 'ml_cnn'
    DEEP_CNN_NO_POOL = 'ml_cnn_no_pool'
    DEEP_FFN = 'ml_ffn'
    NORMALIZED_RNN = 'norm_rnn'
    NORMALIZED_CNN = 'norm_cnn'
    KALMAN_RNN = 'kalman_rnn'
    KALMAN_CNN = 'kalman_cnn'
    KALMAN_NORMALIZED_RNN = 'kal_norm_rnn'
    KALMAN_NORMALIZED_CNN = 'kal_norm_cnn'


class Experiment:
    def __init__(self, connection, graph, batch_size, data_convolut,
                 setting: ExperimentSetting = ExperimentSetting.DEFAULT):
        self.max_instance_length = data_convolut.get_max_instance_length
        self.signature = graph.get_signature_with_timesteps(timesteps=self.max_instance_length)
        self._persistent_signature = self.signature
        self.batch_size = batch_size
        self.setting = setting
        self.configure_experiment(setting=setting)
        self.connection = connection
        self.data_convolut = data_convolut

    @property
    def experiment_name(self):
        name = str(self.data_convolut.instances_train) \
               + '_' + self.setting.value + '_' + self.data_convolut.process_name
        if self.data_convolut.anomalies_flow > 0.0:
            name = name + '_' + str(self.data_convolut.anomalies_flow) + 'flow'
        if self.data_convolut.anomalies_sensor > 0.0:
            name = name + '_' + str(self.data_convolut.anomalies_sensor) + 'sensor'
        return name

    def configure_experiment(self, setting: ExperimentSetting):
        if setting == ExperimentSetting.BASELINE_ID:
            self.set_attribute_preprocessing(Preprocessing.DISABLE)
        elif setting == ExperimentSetting.BASELINE_ATTRIBUTE:
            self.set_id_preprocessing(Preprocessing.DISABLE)
        elif setting == ExperimentSetting.BASELINE_SENSOR:
            self.set_except_sensor_preprocessing(Preprocessing.DISABLE)
        elif setting == ExperimentSetting.DEFAULT:
            return
        elif setting == ExperimentSetting.CONVOLUTION:
            self.set_sensor_preprocessing(Preprocessing.CONVOLUTION)
        elif setting == ExperimentSetting.DEEP_CNN:
            self.set_sensor_preprocessing(Preprocessing.DEEP_CNN)
        elif setting == ExperimentSetting.RNN:
            self.set_sensor_preprocessing(Preprocessing.RNN)
        elif setting == ExperimentSetting.KALMAN:
            self.update_signature_for_added_kalman_spec()
            #self.update_signature_for_deprecated_kalman_spec()
            self.set_sensor_preprocessing(Preprocessing.DISABLE)
        elif setting == ExperimentSetting.HAMPEL:
            self.set_sensor_preprocessing(Preprocessing.HAMPEL)
        elif setting == ExperimentSetting.NORMALIZE:
            self.set_sensor_preprocessing(Preprocessing.NORMALIZE)
        elif setting == ExperimentSetting.DEEP_CNN_NO_POOL:
            self.set_sensor_preprocessing(Preprocessing.DEEP_CNN_NO_POOL)
        elif setting == ExperimentSetting.DEEP_FFN:
            self.set_sensor_preprocessing(Preprocessing.DEEP_FFN)
        elif setting == ExperimentSetting.NORMALIZED_RNN:
            self.set_sensor_preprocessing(Preprocessing.NORMALIZED_RNN)
        elif setting == ExperimentSetting.NORMALIZED_CNN:
            self.set_sensor_preprocessing(Preprocessing.NORMALIZED_CNN)
        elif setting == ExperimentSetting.KALMAN_RNN:
            self.update_signature_for_added_kalman_spec()
            self.set_sensor_preprocessing(Preprocessing.RNN)
        elif setting == ExperimentSetting.KALMAN_CNN:
            self.update_signature_for_added_kalman_spec()
            self.set_sensor_preprocessing(Preprocessing.CONVOLUTION)
        elif setting == ExperimentSetting.KALMAN_NORMALIZED_RNN:
            self.update_signature_for_added_kalman_spec()
            self.set_sensor_preprocessing(Preprocessing.NORMALIZED_RNN)
        elif setting == ExperimentSetting.KALMAN_NORMALIZED_CNN:
            self.update_signature_for_added_kalman_spec()
            self.set_sensor_preprocessing(Preprocessing.NORMALIZED_CNN)
        else:
            raise NotImplementedError

    def update_signature_for_added_kalman_spec(self):
        # Update the Tensor Shape in the Spec if it is altered during the preprocessing
        # This is the case for Kalman since it is very expensive to compute in the model
        offset = 0
        signature = list(self.signature[0])
        for index, spec in enumerate(self._persistent_signature[0]):
            if 'sensor' in spec.name:
                signature[index + offset:index + offset] = [BinetTensorSpec(shape=(spec.shape[0], 1), dtype=spec.dtype,
                                                                            name=spec.name.replace("sensor", "kalman"),
                                                                            preprocessing=Preprocessing.DEFAULT,
                                                                            sensor_anomalies=spec.sensor_anomalies)]
                offset += 1
        self.signature = (tuple(signature), self.signature[1])

    def update_signature_for_deprecated_kalman_spec(self):
        # Update the Tensor Shape in the Spec if it is altered during the preprocessing
        # This is the case for Kalman since it is very expensive to compute in the model
        signature = list(self.signature[0])
        for index, spec in enumerate(self._persistent_signature[0]):
            if 'sensor' in spec.name:
                signature[index:index] = [BinetTensorSpec(shape=(spec.shape[0], 1), dtype=spec.dtype,
                                                          name=spec.name,
                                                          preprocessing=Preprocessing.DEFAULT,
                                                          sensor_anomalies=spec.sensor_anomalies)]
        self.signature = (tuple(signature), self.signature[1])

    def set_attribute_preprocessing(self, preprocessing: Preprocessing):
        for element in self.signature[0][1:]:
            element.set_preprocessing(preprocessing)

    def set_sensor_preprocessing(self, preprocessing: Preprocessing):
        for element in self.signature[0]:
            if 'sensor' in element.name:
                element.set_preprocessing(preprocessing)

    def set_except_sensor_preprocessing(self, preprocessing: Preprocessing):
        for element in self.signature[0]:
            if 'sensor' not in element.name:
                element.set_preprocessing(preprocessing)

    def set_id_preprocessing(self, preprocessing: Preprocessing):
        self.signature[0][0].set_preprocessing(preprocessing)

    def get_model_summary(self, model: tf.keras.Model) -> str:
        string_list = []
        model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)

    def save_model_image(self):
        model = PredictionModel(self.signature, batch_size=self.batch_size, window_size=self.max_instance_length).model
        tf.keras.utils.plot_model(
            model,
            to_file="model.png",
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
        )
        texcode = dot2tex.dot2tex(str(tf.keras.utils.model_to_dot(
            model,
            show_shapes=False,
            show_dtype=False,
            show_layer_names=False,
            rankdir="LR",
            expand_nested=False,
            dpi=96,
            subgraph=False,
            layer_range=None,
        )), format='pgf', crop=True, figonly=True)
        with open(get_assets_path() / 'SystemArchitecture' / "model_vis.pgf", 'w') as f:
            f.write(texcode)

    def train_model(self):
        model = PredictionModel(self.signature, batch_size=self.batch_size).model
        print(model.summary())
        # configure early stopping
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

        log_dir = get_log_output_path() / 'fit' / 'tensorboard' / self.experiment_name / datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        epochs = 20
        checkpoint_path = get_log_output_path() / 'fit' / 'checkpoints' / self.experiment_name

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.8, min_lr=1e-6)
        # utility callback that displays training curves
        plot_training = PlotTraining(batch_size=self.batch_size, sample_rate=10, zoom=1)
        # lr schedule callback
        # lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)

        #model.load_weights(checkpoint_path)

        start = datetime.now()
        # print(self.dataset.train)

        model.fit(self.data_convolut.train.data,
                  epochs=epochs,
                  # We pass some validation for
                  # monitoring validation loss and metrics
                  # at the end of each epoch
                  validation_data=self.data_convolut.dev.data,
                  callbacks=[
                      tensorboard_callback,
                      early_stopping,
                      cp_callback,
                      reduce_on_plateau]
                  )
        model.save(get_model_output_path() / self.experiment_name)
        end = datetime.now()

        return model

    def print_model_latex(self):
        path = get_model_output_path() / self.experiment_name
        print('load %s' % path)
        model = tf.keras.models.load_model(path)
        table = pd.DataFrame(columns=["Name", "Type", "Shape"])
        for layer in model.layers:
            table = table.append({"Name": layer.name, "Type": layer.__class__.__name__, "Shape": layer.output_shape},
                                 ignore_index=True)
        print(table.to_latex())


    def run(self, only_plots=False):
        try:
            path = get_model_output_path() / self.experiment_name
            print('load %s'%path)
            model = tf.keras.models.load_model(path)
        except:
            model = self.train_model()
        self.evaluate_model(model=model, only_plots=only_plots)

    def evaluate_model(self, model, only_plots=False):

        i_layer = tf.keras.layers.StringLookup(vocabulary=self.data_convolut.graph.node_names, invert=True)

        correct_labels = []
        predicted_labels = []
        for input, target in self.data_convolut.test.data:
            predictions = model.predict(input)
            assert predictions.shape == target.shape, 'The model does not fit the data'
            for index in range(predictions.shape[0]):
                correct_labels.append(str(i_layer([np.argmax(target[index])]).numpy()[0].decode('UTF-8')))
                predicted_labels.append(str(i_layer([np.argmax(predictions[index])]).numpy()[0].decode('UTF-8')))

        label = list(set(correct_labels))
        label.sort()
        cm = confusion_matrix(correct_labels, predicted_labels, normalize='all', labels=label)

        df_cm = pd.DataFrame(cm, index=label, columns=label)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, fmt='.2f', cmap=sn.color_palette("rocket_r", as_cmap=True))
        #plt.title('Normalized Confusion Matrix of ' + self.experiment_name, fontsize=22)
        plt.ylabel('True Activities', fontsize=18)
        plt.xlabel('Predicted Activities', fontsize=18)
        plt.tight_layout()
        path = get_assets_path() / 'Experiments' / self.experiment_name
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path / "confusion_matrix_all.pgf", format='pgf')
        plt.clf()

        print(f'Cohen’s kappa: {cohen_kappa_score(correct_labels, predicted_labels):.2f}')
        print(f'Accuracy: {accuracy_score(correct_labels, predicted_labels):.2f}')
        print(f'F1: {f1_score(correct_labels, predicted_labels, average="weighted"):.2f}')
        print(
            f'Precision: {precision_score(correct_labels, predicted_labels, average="weighted", zero_division=0):.2f}')
        print(f'Recall: {recall_score(correct_labels, predicted_labels, average="weighted", zero_division=0):.2f}')
        if not only_plots:
            create_experiment(conn=self.connection,
                              preprocessing=self.setting.value,
                              train_dataset=self.data_convolut.train.id,
                              test_dataset=self.data_convolut.test.id,
                              filename=str(get_model_output_path() / self.experiment_name),
                              training_duration=0,
                              hyperparameter=self.get_model_summary(model),
                              epochs=0,
                              accuracy=accuracy_score(correct_labels, predicted_labels),
                              kappa=cohen_kappa_score(correct_labels, predicted_labels),
                              precision=precision_score(correct_labels, predicted_labels, average="weighted",
                                                        zero_division=0),
                              recall=recall_score(correct_labels, predicted_labels, average="weighted",
                                                  zero_division=0),
                              f1=f1_score(correct_labels, predicted_labels, average="weighted"))
        # print(f'Area Under the Receiver Operating Characteristic Curve: {roc_auc_score(correct_labels, predicted_labels, average="micro", multi_class="ovr"):.2f}')
        if self.data_convolut.test_decisions:
            prediction_dict = defaultdict(list)
            label_dict = defaultdict(list)
            correct_labels = []
            predicted_labels = []
            for input, target in self.data_convolut.test_decisions.data:
                predictions = model.predict(input)
                assert predictions.shape == target.shape, 'The model does not fit the data'
                for index in range(predictions.shape[0]):
                    key_tmp = i_layer([np.argmax(input[0][index][-1])]).numpy()[0].decode('UTF-8')
                    target_tmp = i_layer([np.argmax(target[index])]).numpy()[0].decode('UTF-8')
                    predictions_tmp = i_layer([np.argmax(predictions[index])]).numpy()[0].decode('UTF-8')
                    prediction_dict[key_tmp].append(predictions_tmp)
                    label_dict[key_tmp].append(target_tmp)
                    predicted_labels.append(predictions_tmp)
                    correct_labels.append(target_tmp)
            print(f'Cohen’s kappa: {cohen_kappa_score(correct_labels, predicted_labels):.2f}')
            print(f'Accuracy: {accuracy_score(correct_labels, predicted_labels):.2f}')
            print(f'F1: {f1_score(correct_labels, predicted_labels, average="weighted"):.2f}')
            print(
                f'Precision: {precision_score(correct_labels, predicted_labels, average="weighted", zero_division=0):.2f}')
            print(f'Recall: {recall_score(correct_labels, predicted_labels, average="weighted", zero_division=0):.2f}')
            if not only_plots:
                create_experiment(conn=self.connection,
                                  preprocessing=self.setting.value,
                                  train_dataset=self.data_convolut.train.id,
                                  test_dataset=self.data_convolut.test_decisions.id,
                                  filename=str(get_model_output_path() / self.experiment_name),
                                  training_duration=0,
                                  hyperparameter=self.get_model_summary(model),
                                  epochs=0,
                                  accuracy=accuracy_score(correct_labels, predicted_labels),
                                  kappa=cohen_kappa_score(correct_labels, predicted_labels),
                                  precision=precision_score(correct_labels, predicted_labels, average="weighted",
                                                            zero_division=0),
                                  recall=recall_score(correct_labels, predicted_labels, average="weighted",
                                                      zero_division=0),
                                  f1=f1_score(correct_labels, predicted_labels, average="weighted"))

            node_to_decision = {'prescribe': 'HD4', 'diagnosis': 'HD3', 'nurse': 'HD2',
                                'reception': 'HD1', 'extract': 'LD3', 'material_test': 'LD2', 'wash': 'LD1'}
            for key in prediction_dict.keys():
                label = list(set(label_dict[key]))
                label.sort()
                cm = confusion_matrix(label_dict[key], prediction_dict[key], normalize='all', labels=label)
                df_cm = pd.DataFrame(cm, index=label, columns=label)
                plt.figure(figsize=(4, 4))
                sn.heatmap(df_cm, annot=True, fmt='.2f', cmap=sn.color_palette("rocket_r", as_cmap=True), cbar=False,
                           annot_kws={"size": 10})
                # plt.title(node_to_decision[key], fontsize=15)
                plt.ylabel('True Activities', fontsize=12)
                plt.xlabel('Predicted Activities', fontsize=12)
                plt.tight_layout()
                path = get_assets_path() / 'Experiments' / self.experiment_name
                if not os.path.exists(path):
                    os.mkdir(path)
                filename = 'confusion_matrix_' + node_to_decision[key] + '.pgf'
                plt.savefig(path / filename, format='pgf')
                plt.clf()
