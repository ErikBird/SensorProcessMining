import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.stats import pearsonr
import tensorflow as tf
from ProcessSimulator.src.simulator.process_model import Preprocessing


def preprocess_step(step_data, signature):

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
        return pred_state

    output = []
    for attribute, spec in zip(step_data, signature[0]):
        if 'sensor' in spec.name:
            if tf.is_tensor(attribute):
                attribute = attribute.numpy()
            if not isinstance(attribute, np.ndarray):
                attribute = np.array(attribute)
            if attribute.nonzero()[0].size != 0:
                '''
                if spec.preprocessing == Preprocessing.HAMPEL:
                    filtered_x = hampel(tf.squeeze(attribute).numpy())
                    filtered_x = tf.squeeze(filtered_x)
                elif spec.preprocessing == Preprocessing.KALMAN:'''
                filtered_x = np.array(Kalman1D(attribute))
                filtered_x = tf.squeeze(filtered_x)
                # else:raise NotImplementedError
                corr, p_value = pearsonr(x=attribute, y=filtered_x.numpy())
                if not np.isnan(corr):
                    output.append([corr])
                    output.append(attribute)
                else:
                    output.append([1.0])
                    output.append(attribute)
            else:
                output.append([1.0])
                output.append(attribute)
        else:
            output.append(attribute)
    return output
