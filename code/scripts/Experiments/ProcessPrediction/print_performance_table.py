from ProcessSimulator.src.processes import LaboratorySimulator
from ProcessSimulator.src.processes.hospital_process.hospital_process import HospitalSimulator
from ProcessSimulator.src.simulator.process_model import DatasetStage
from utils import get_database_connection, get_assets_path, get_matplotlib_size, get_color_hex_list
import matplotlib.pyplot as plt
import pandas as pd

conn = get_database_connection()
laboratory_process = LaboratorySimulator()
hospital_process = HospitalSimulator()


def print_latex(graph, stage, instances, flow_anomalies=0.0, sensor_anomalies=0.0):
    df = pd.read_sql_query("""SELECT
                                    preprocessing,
                                    accuracy,
                                    kappa,
                                    precision,
                                    recall,
                                    f1
                                FROM
                                    experiment
                                LEFT JOIN dataset as test_dataset ON experiment.test_dataset = test_dataset.id
                                LEFT JOIN dataset as train_dataset ON experiment.train_dataset = train_dataset.id
                                WHERE test_dataset.generation_setting=:stage 
                                AND train_dataset.num_instances=:train_instances
                                AND test_dataset.process=:process
                                AND train_dataset.anomalies_flow=:flow_anomalies
                                AND train_dataset.anomalies_sensor=:sensor_anomalies
                                ORDER BY kappa;""",
                           conn,
                           params={"process": graph.__name__, "stage": stage.value,
                                   'train_instances': instances, 'flow_anomalies': flow_anomalies,
                                   'sensor_anomalies':sensor_anomalies})
    name = graph.__name__.replace('Simulator', '')
    caption = 'Performance of the model with different pre-processing configurations evaluated on '
    if stage == DatasetStage.TEST_DECISION:
        caption += '\\textbf{the decisions} of '
    caption += name + str(instances) + '.'
    label = 'tab:performance_' + name + str(instances) + '_' + stage.value
    advanced = ['norm_rnn', 'norm_cnn', 'kalman_rnn', 'kalman_cnn', 'kal_norm_rnn', 'kal_norm_cnn']
    df = df.query("preprocessing not in @advanced")
    df.sort_values(by=['preprocessing'], inplace=True, ascending=False)
    print(df.to_latex(index=False, caption=caption, label=label, float_format="%.4f"))


for process in [laboratory_process, hospital_process]:
    for stage in [DatasetStage.TEST, DatasetStage.TEST_DECISION]:
        for size in [10000]:#, 10000]:
            print_latex(process, stage, size, sensor_anomalies=0.3, flow_anomalies=0.3)
