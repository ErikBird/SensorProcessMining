from ProcessSimulator.src.processes import LaboratorySimulator
from ProcessSimulator.src.processes.hospital_process.hospital_process import HospitalSimulator
from ProcessSimulator.src.simulator.process_model import DatasetStage
from utils import get_database_connection, get_assets_path, get_matplotlib_size, get_color_hex_list
import matplotlib.pyplot as plt
import pandas as pd

conn = get_database_connection()
laboratory_process = LaboratorySimulator()
hospital_process = HospitalSimulator()


def print_latex(graph, instances, only_first=False):
    if only_first:
        anomalies = 'only_first'
    else:
        anomalies = 'all'

    df = pd.read_sql_query("""SELECT
    graph,
                                    preprocessing,
                                    accuracy,
                                    kappa,
                                    precision,
                                    recall,
                                    f1
                                FROM
                                    anomaly
                                WHERE anomalies =:anomalies
                               AND graph =:graph;""",
                           conn,
                           params={"graph": str(instances)+graph.__name__,
                           'anomalies': anomalies})
    name = graph.__name__.replace('Simulator', '')
    caption = 'Performance of the model with different pre-processing configurations evaluated on '

    caption += name + str(instances) + '.'
    #label = 'tab:performance_' + name + str(instances) + '_' + stage.value
    #advanced = ['norm_rnn', 'norm_cnn', 'kalman_rnn', 'kalman_cnn', 'kal_norm_rnn', 'kal_norm_cnn']
    #df = df.query("preprocessing not in @advanced")
    df.sort_values(by=['preprocessing'], inplace=True, ascending=False)
    print(df.to_latex(index=False, caption=caption, float_format="%.4f"))#label=label


for process in [laboratory_process, hospital_process]:
        for size in [10000]:#, 10000]:
            print_latex(process, size, only_first=False)
