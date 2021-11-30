import tensorflow as tf

from ProcessSimulator.src.processes import HospitalSimulator, LaboratorySimulator, WindowSimulator
from database import initialize_table, create_dataset, create_experiment
from dataset import DataConvolut
from experiment import ExperimentSetting, Experiment
from utils import get_dataset_output_path, get_log_output_path, get_database_connection
import gc

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def test_series(connection):
    for train_size, dev_size, test_size in [ (6000, 1000, 2000)]:  # ]:
        for sensor_anomalies, flow_anomalies in [(0.0, 0.0)]:  # (0.3, 0.0), (0.0, 0.3),
            for graph in [
                LaboratorySimulator,
                #HospitalSimulator
            ]:
                graph = graph()
                print(train_size)
                print('flow_anomalies %s, sensor_anomalies %s' % (flow_anomalies, sensor_anomalies))
                data_convolut = DataConvolut(connection=connection, instances_train=train_size,
                                             instances_dev=dev_size,
                                             instances_test=test_size,
                                             graph=graph,
                                             create_only_decisions=True,
                                             batch_size=128,
                                             anomalies_flow=flow_anomalies,
                                             anomalies_sensor=sensor_anomalies,
                                             setting=ExperimentSetting.KALMAN)
                for setting in [
                    # ExperimentSetting.CONVOLUTION,
                    # ExperimentSetting.DEEP_CNN,
                    # ExperimentSetting.RNN,
                    # ExperimentSetting.DEFAULT,
                    # ExperimentSetting.NORMALIZE,
                    # ExperimentSetting.BASELINE_SENSOR,
                    #ExperimentSetting.BASELINE_ID,
                    # ExperimentSetting.BASELINE_ATTRIBUTE,
                    # ExperimentSetting.DEEP_CNN_NO_POOL,
                    # ExperimentSetting.DEEP_FFN,
                    # ExperimentSetting.NORMALIZED_RNN,
                    # ExperimentSetting.NORMALIZED_CNN,
                    ExperimentSetting.KALMAN,
                    # ExperimentSetting.KALMAN_RNN,
                    # ExperimentSetting.KALMAN_CNN,
                    # ExperimentSetting.KALMAN_NORMALIZED_RNN,
                    # ExperimentSetting.KALMAN_NORMALIZED_CNN,

                ]:
                    print('ExperimentSetting: %s' % setting.value)
                    experiment = Experiment(connection=connection, graph=graph, data_convolut=data_convolut,
                                            setting=setting, batch_size=128)

                    experiment.run(only_plots=True)  #
                    #experiment.print_model_latex()
                    del experiment

                del data_convolut
                gc.collect()


if __name__ == "__main__":
    import logging

    conn = get_database_connection()
    initialize_table(conn)

    tf.get_logger().setLevel(logging.ERROR)
    test_series(conn)
