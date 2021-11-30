import copy

import numpy as np
from keras.layers import Lambda, Conv1D
from pykalman import KalmanFilter
from scipy.stats import pearsonr
from tensorflow.python.layers.base import Layer
from tensorflow.python.util.tf_export import keras_export
import tensorflow_probability as tfp
from ProcessSimulator.src.simulator.process_model import Preprocessing
import tensorflow as tf
import tensorflow_addons as tfa


@tf.function  # (input_signature=(tf.TensorSpec(shape=(140), dtype=tf.float32),))
def my_preprocess_tf(x):
    size = x.shape[0]
    x = tf.reshape(x, shape=(size,))
    Q = tf.constant(1e-5)  # process variance

    # allocate space for arrays
    xhat = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # a posteri estimate of x
    P = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # a posteri error estimate
    xhatminus = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # a priori estimate of x
    Pminus = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # a priori error estimate
    K = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # gain or blending factor

    R = tf.constant(0.1 ** 2)  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat = xhat.write(0, x[0])  #
    P = P.write(0, 1.0)

    for k in tf.range(1, size):
        # time update
        xhatminus = xhatminus.write(k, xhat.read(k - 1))
        Pminus = Pminus.write(k, P.read(k - 1) + Q)

        # measurement update
        pminus_k = Pminus.read(k)
        xhatminus_k = xhatminus.read(k)
        K_k = pminus_k / (pminus_k + R)
        K = K.write(k, K_k)
        delta = x[k] - xhatminus_k
        mmm = xhatminus_k + K_k
        prediction = tf.multiply(mmm, delta)
        xhat = xhat.write(k, prediction)
        P = P.write(k, (1 - K_k) * pminus_k)

    corr = tfp.stats.correlation(x, xhat.stack(), sample_axis=0, event_axis=None)
    return tf.reshape(corr, shape=())


@tf.function
def kalman_filter_tf(input):
    size = tf.shape(input)[0]
    time_t = tf.reshape(input, shape=(size, 1))
    filter_dimension = 1  # 1D Kalman
    step_std = 1.0
    noise_std = 2.4
    diag = tf.reshape(time_t[0] ** 2, shape=(1,))
    _, filtered_means, filtered_covs, _, _, _, _ = tfp.distributions.LinearGaussianStateSpaceModel(
        num_timesteps=size,
        transition_matrix=tf.linalg.LinearOperatorIdentity(filter_dimension),
        transition_noise=tfp.distributions.MultivariateNormalDiag(
            scale_diag=step_std ** 2 * tf.ones([filter_dimension])),
        observation_matrix=tf.linalg.LinearOperatorIdentity(filter_dimension),
        observation_noise=tfp.distributions.MultivariateNormalDiag(
            scale_diag=noise_std ** 2 * tf.ones([filter_dimension])),
        initial_state_prior=tfp.distributions.MultivariateNormalDiag(loc=None,
                                                                     scale_diag=diag)).forward_filter(time_t)
    # _, filtered_means, filtered_covs, _, _, _, _ = model.forward_filter(time_t)
    corr = tfp.stats.correlation(time_t, filtered_means, sample_axis=0, event_axis=None)
    return tf.reshape(corr, shape=())


@tf.function  # (input_signature=(tf.TensorSpec(shape=(128, 7, 140), dtype=tf.float32),))
def apply_on_batch_timestamp(input):
    kalman = my_preprocess_tf.get_concrete_function(tf.TensorSpec(shape=(None), dtype=tf.float32))
    # https://www.tensorflow.org/guide/function#loops
    batch_size, time_size, feature_size = input.shape
    batch_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for batch_index in tf.range(batch_size):
        time_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for time_index in tf.range(time_size):
            sensor_data = input[batch_index, time_index, :]
            result = kalman(sensor_data)
            time_ta = time_ta.write(time_index, result)
        batch_ta = batch_ta.write(batch_index, time_ta.stack())
    return tf.reshape(tf.cast(batch_ta.stack(), dtype=tf.float32), shape=(batch_size, time_size, 1))


def kalman_batch_timestep_sensor(input):
    print("Tracing!")  # An eager-only side effect.

    # https://www.tensorflow.org/guide/function#loops
    # batch_size, time_size, feature_size = input.shape
    batch_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for batch_index in tf.range(input.shape[0]):
        time_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for time_index in tf.range(input.shape[1]):
            sensor_data = input[batch_index, time_index, :]
            # tf.print(sensor_data.shape)
            # size = sensor_data.shape[0]
            x = tf.reshape(sensor_data, shape=(input.shape[2],))
            Q = tf.constant(1e-5)  # process variance

            # allocate space for arrays
            xhat = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # a posteri estimate of x
            P = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # a posteri error estimate
            xhatminus = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # a priori estimate of x
            Pminus = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # a priori error estimate
            K = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # gain or blending factor

            R = tf.constant(0.1 ** 2)  # estimate of measurement variance, change to see effect

            # intial guesses
            xhat = xhat.write(0, x[0])  #
            P = P.write(0, 1.0)

            for k in tf.range(1, input.shape[2]):
                # time update
                xhatminus = xhatminus.write(k, xhat.read(k - 1))
                Pminus = Pminus.write(k, P.read(k - 1) + Q)

                # measurement update
                pminus_k = Pminus.read(k)
                xhatminus_k = xhatminus.read(k)
                K_k = pminus_k / (pminus_k + R)
                K = K.write(k, K_k)
                delta = x[k] - xhatminus_k
                mmm = xhatminus_k + K_k
                prediction = tf.multiply(mmm, delta)
                xhat = xhat.write(k, prediction)
                P = P.write(k, (1 - K_k) * pminus_k)

            corr = tfp.stats.correlation(x, xhat.stack(), sample_axis=0, event_axis=None)
            result = tf.reshape(corr, shape=())
            time_ta = time_ta.write(time_index, result)
        batch_ta = batch_ta.write(batch_index, time_ta.stack())
    return tf.reshape(tf.cast(batch_ta.stack(), dtype=tf.float32), shape=(input.shape[0], input.shape[1], 1))


def Kalman1D(observations, damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, state_cov = kf.smooth(observations)
    corr, p_value = pearsonr(x=np.squeeze(pred_state), y=observations)
    if not np.isnan(corr):
        return np.float32(corr)
    else:
        return np.float32(1.0)


@tf.function
def gold_kalman_batch_timestep_sensor(input):
    print("Tracing!")  # An eager-only side effect.

    # https://www.tensorflow.org/guide/function#loops
    # batch_size, time_size, feature_size = input.shape
    batch_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for batch_index in tf.range(input.shape[0]):
        time_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for time_index in tf.range(input.shape[1]):
            sensor_data = input[batch_index, time_index, :]
            corr = tf.numpy_function(Kalman1D, [sensor_data], tf.float32)
            result = tf.reshape(corr, shape=())
            time_ta = time_ta.write(time_index, result)
        batch_ta = batch_ta.write(batch_index, time_ta.stack())
    return tf.reshape(tf.cast(batch_ta.stack(), dtype=tf.float32), shape=(input.shape[0], input.shape[1], 1))


'''
class Kalman1DLayer(Layer):
    def __init__(self, name=None, **kwargs):
        super(Kalman1DLayer, self).__init__(name=name, **kwargs)
        self.preprocess = kalman_batch_timestep_sensor.get_concrete_function(tf.TensorSpec(shape=(128, 7, 186),
                                                                                           dtype=tf.float32))

    # def compute_output_shape(self, input_shape):
    #    return (input_shape[0], input_shape[1], 1)

    def get_config(self):
        return super(Kalman1DLayer, self).get_config()

    def call(self, input):
        return self.preprocess(input)

'''


class PredictionModel(object):
    def __init__(self, signature, batch_size: int):
        # A given Batch size is needed since the Kalman Filter Preprocessing layer unpacks the tensor
        # unpack is compiled into "tensor-in/tensor-out" ops during graph construction time
        self._batch_size = batch_size
        self._model = self.get_model_from_signature(signature)
        # self.kalman = apply_on_batch_timestamp.get_concrete_function(tf.TensorSpec(shape=(128, 7, 140),
        # dtype=tf.float32))

    @property
    def model(self):
        return self._model

    def summary(self):
        return self._model.summary()

    def fit(self, **kwargs):
        from SensorNet.src.util import PlotTraining, lr_decay

        # tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/%s' % NAME)

        # utility callback that displays training curves
        # plot_training = PlotTraining(batch_size=BATCH_SIZE, sample_rate=10, zoom=1)
        # lr schedule callback
        # lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)

        print("Fit model on training data")

        history = self._model.fit(epochs=5, **kwargs)
        # dataset,
        # epochs=5,
        # validation_data=test_dataset,
        # callbacks=[plot_training, tensorboard]  # , lr_decay_callback]
        # )
        return history

    def evaluate(self, test_dataset):
        return self._model.evaluate(test_dataset)
        # loss, acc = model.evaluate(test_dataset, verbose=2)
        # print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    def get_model_from_signature(self, signature):
        # [batch, time, features] => [batch, time, lstm_units]
        inputs = []
        concat_input = []
        custom_layer = {}  # shape -> layer

        for spec in signature[0]:
            '''
            if spec.preprocessing == Preprocessing.ID:
                print(spec.shape)
                input_layer = tf.keras.Input(shape=spec.shape, name= spec.name, dtype=spec.dtype)
                print(input_layer.shape)
                # Use IntegerLookup to build an index of the feature values
                indexer = tf.keras.layers.StringLookup()#IntegerLookup
                indexer.adapt(model.node_names)
            
                node_names = model.node_names
                #indexer.adapt(node_names)
            
                # Use CategoryEncoding to encode the integer indices to a one-hot vector
                encoder = tf.keras.layers.CategoryEncoding(num_tokens=len(node_names)+1,output_mode="one_hot")
            
                #one_hot = encoder(indexer(input_layer))
            
                one_hot = tf.keras.layers.Lambda(lambda x: encoder(indexer(x)))(input_layer)
                print(one_hot.shape)
                inputs.append(input_layer)
                concat_input.append(tf.cast(one_hot, dtype=tf.float32))
            else:'''
            if spec.preprocessing == Preprocessing.DISABLE:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                inputs.append(input_layer)
            elif spec.preprocessing == Preprocessing.CONVOLUTION:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                conv_layer = tf.keras.layers.Conv1D(10, 5, activation='relu', padding='same', input_shape=spec.shape)(
                    input_layer)
                inputs.append(input_layer)
                concat_input.append(tf.cast(conv_layer, dtype=tf.float32))
            elif spec.preprocessing == Preprocessing.DEEP_CNN:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                conv_layer = tf.keras.layers.Conv1D(spec.shape[1] / 3, 3, activation='relu', padding='same',
                                                    input_shape=spec.shape)(input_layer)
                pool = tf.keras.layers.MaxPooling1D(3, data_format='channels_first', )(conv_layer)
                conv_layer = tf.keras.layers.Conv1D(spec.shape[1] / 6, 3, activation='relu', padding='same',
                                                    input_shape=spec.shape)(
                    pool)
                pool = tf.keras.layers.MaxPooling1D(2, data_format='channels_first')(conv_layer)
                conv_layer = tf.keras.layers.Conv1D(spec.shape[1] / 12, 3, activation='relu', padding='same',
                                                    input_shape=spec.shape)(
                    pool)
                pool = tf.keras.layers.MaxPooling1D(2, data_format='channels_first')(conv_layer)
                inputs.append(input_layer)
                concat_input.append(tf.cast(pool, dtype=tf.float32))
            elif spec.preprocessing == Preprocessing.DEEP_CNN_NO_POOL:
                print('spec shape: %s' % spec.shape)
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                conv_layer = tf.keras.layers.Conv1D(spec.shape[1] / 3, 3, activation='relu', padding='same',
                                                    input_shape=spec.shape)(input_layer)
                conv_layer = tf.keras.layers.Conv1D(spec.shape[1] / 6, 3, activation='relu', padding='same',
                                                    input_shape=spec.shape)(
                    conv_layer)
                conv_layer = tf.keras.layers.Conv1D(spec.shape[1] / 12, 3, activation='relu', padding='same',
                                                    input_shape=spec.shape)(
                    conv_layer)
                inputs.append(input_layer)
                concat_input.append(tf.cast(conv_layer, dtype=tf.float32))
            elif spec.preprocessing == Preprocessing.DEEP_FFN:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                dense = tf.keras.layers.Dense(units=50, activation='relu')(input_layer)
                dense = tf.keras.layers.Dense(units=50, activation='relu')(dense)
                inputs.append(input_layer)
                concat_input.append(tf.cast(dense, dtype=tf.float32))
            elif spec.preprocessing == Preprocessing.RNN:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                rnn_layer = tf.keras.layers.GRU(10, return_sequences=True)(input_layer)
                inputs.append(input_layer)
                concat_input.append(tf.cast(rnn_layer, dtype=tf.float32))
            elif spec.preprocessing == Preprocessing.NORMALIZE:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)

                mean = Lambda(lambda x: tf.expand_dims(tf.experimental.numpy.mean(x, axis=2), axis=2))(input_layer)
                normalization = tf.keras.layers.LayerNormalization(axis=2)(input_layer)

                inputs.append(input_layer)
                concat_input.append(mean)
                concat_input.append(normalization)
            elif spec.preprocessing == Preprocessing.NORMALIZED_CNN:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                mean = Lambda(lambda x: tf.expand_dims(tf.experimental.numpy.mean(x, axis=2), axis=2))(input_layer)
                normalization = tf.keras.layers.LayerNormalization(axis=2)(input_layer)
                conv_layer = tf.keras.layers.Conv1D(10, 5, activation='relu', padding='same', input_shape=spec.shape)(
                    normalization)
                inputs.append(input_layer)
                concat_input.append(mean)
                concat_input.append(tf.cast(conv_layer, dtype=tf.float32))
            elif spec.preprocessing == Preprocessing.NORMALIZED_RNN:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                mean = Lambda(lambda x: tf.expand_dims(tf.experimental.numpy.mean(x, axis=2), axis=2))(input_layer)
                normalization = tf.keras.layers.LayerNormalization(axis=2)(input_layer)
                rnn_layer = tf.keras.layers.GRU(10, return_sequences=True)(normalization)
                inputs.append(input_layer)
                concat_input.append(mean)
                concat_input.append(tf.cast(rnn_layer, dtype=tf.float32))
            elif spec.preprocessing == Preprocessing.KALMAN:
                tf.print(spec.shape)
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32,
                                             batch_size=self._batch_size)
                kalman_layer = Lambda(lambda x: gold_kalman_batch_timestep_sensor(x))(input_layer)
                inputs.append(input_layer)
                concat_input.append(kalman_layer)
            else:
                input_layer = tf.keras.Input(shape=spec.shape, name=spec.name, dtype=tf.float32)
                inputs.append(input_layer)
                concat_input.append(input_layer)

        if len(concat_input) > 1:
            # Avoids ValueError: A merge layer should be called on a list of inputs.
            # When loading the saved model
            concat = tf.keras.layers.Concatenate(axis=2)(concat_input)
        else:
            concat = concat_input[0]
        encoder = tf.keras.layers.GRU(300, return_sequences=True, activation='tanh', name=f'encoder')(concat)
        encoder_normalization = tf.keras.layers.BatchNormalization(name='encoder_normalization')(encoder)
        decoder = tf.keras.layers.GRU(200, return_sequences=True, activation='tanh', name=f'decoder')(
            encoder_normalization)
        decoder_normalization = tf.keras.layers.BatchNormalization(name=f'decoder_normalization')(decoder)
        flatten_output = tf.keras.layers.Flatten()(decoder_normalization)
        outputs = tf.keras.layers.Dense(units=signature[1].shape[0], activation='softmax')(flatten_output)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        from keras.optimizer_v2.adam import Adam

        model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.Accuracy(),
                     tf.keras.metrics.AUC(),
                     tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall()]
            # tfa.metrics.F1Score()],
        )

        return model
