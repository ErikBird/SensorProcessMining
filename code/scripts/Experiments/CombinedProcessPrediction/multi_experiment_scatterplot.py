from ProcessSimulator.src.processes import LaboratorySimulator
from ProcessSimulator.src.processes.hospital_process.hospital_process import HospitalSimulator
from ProcessSimulator.src.simulator.process_model import DatasetStage
from utils import get_database_connection, get_assets_path, get_matplotlib_size, get_color_hex_list
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import progressbar
import numpy as np
import matplotlib.colors


def plot_scatter(connection, instances, only_decision=False, flow_anomalies=0.0,
                 sensor_anomalies=0.0, save_legend=False):
    laboratory_process = LaboratorySimulator()
    hospital_process = HospitalSimulator()

    if only_decision:
        stage = DatasetStage.TEST_DECISION.value
    else:
        stage = DatasetStage.TEST.value
    laboratory_df = pd.read_sql_query("""SELECT
                                    kappa,
                                    preprocessing
                                FROM
                                    experiment
                                LEFT JOIN dataset as test_dataset ON experiment.test_dataset = test_dataset.id
                                LEFT JOIN dataset as train_dataset ON experiment.train_dataset = train_dataset.id
                                WHERE test_dataset.process=:name AND test_dataset.generation_setting=:stage 
                                AND train_dataset.num_instances=:num_instances 
                                AND train_dataset.anomalies_flow=:flow_anomalies
                                AND train_dataset.anomalies_sensor=:sensor_anomalies;""",
                                      connection,
                                      params={"name": laboratory_process.__name__, "stage": stage,
                                              "num_instances": instances, 'flow_anomalies': flow_anomalies,
                                              'sensor_anomalies': sensor_anomalies})

    hospital_df = pd.read_sql_query("""SELECT
                                    kappa,
                                    preprocessing
                                FROM
                                    experiment
                                LEFT JOIN dataset as test_dataset ON experiment.test_dataset = test_dataset.id
                                LEFT JOIN dataset as train_dataset ON experiment.train_dataset = train_dataset.id
                                WHERE test_dataset.process=:name AND test_dataset.generation_setting=:stage 
                                AND train_dataset.num_instances=:num_instances
                                AND train_dataset.anomalies_flow=:flow_anomalies
                                AND train_dataset.anomalies_sensor=:sensor_anomalies;""",
                                    connection,
                                    params={"name": hospital_process.__name__, "stage": stage,
                                            "num_instances": instances, 'flow_anomalies': flow_anomalies,
                                            'sensor_anomalies': sensor_anomalies})

    df = pd.merge(laboratory_df, hospital_df, on='preprocessing', how='left')
    df.sort_values(by=['preprocessing'], inplace=True, ascending=False)
    fig, ax = plt.subplots(1, 1,
                           figsize=get_matplotlib_size(418.25555))  # 1, 1, figsize=get_matplotlib_size(418.25555))

    for index, row in df.iterrows():
        advanced = ['norm_rnn', 'norm_cnn', 'kalman_rnn', 'kalman_cnn', 'kal_norm_rnn', 'kal_norm_cnn']
        #if row['preprocessing'] in advanced:
        marker = "x"
        if 'baseline' in row['preprocessing']:
            marker = '.'
        if row['preprocessing'] in advanced:
            marker = '+'
        ax.scatter([row['kappa_x']], [row['kappa_y']], label=row['preprocessing'], alpha=1.0,
                   marker=marker)

    ax.set_xlabel(r'Kappa Laboratory Dataset')
    ax.set_ylabel(r'Kappa Hospital Dataset')

    #ax.legend(bbox_to_anchor=(1.05, 1), prop={'size': 6})
    decision = ''
    if only_decision:
        decision = '_decision'
    flow_name = ''
    if flow_anomalies > 0.0:
        flow_name += '_' + str(flow_anomalies) + 'flow_'
    if sensor_anomalies > 0.0:
        flow_name += '_' + str(sensor_anomalies) + 'sensor_'
    name = str(instances) + decision + flow_name + ".pgf"

    #ax.set_title(name)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    fig.tight_layout()
    fig.savefig(get_assets_path() / 'Scatterplots' / 'AllProcessPrediction' / name, format='pgf')
    plt.show()

    if save_legend:
        figsize = (2, 3)
        fig_leg = plt.figure(figsize=figsize)
        ax_leg = fig_leg.add_subplot(111)
        # add the legend from the previous axes
        ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')
        # hide the axes frame and the x/y labels
        ax_leg.axis('off')
        fig_leg.tight_layout()
        fig_leg.savefig(get_assets_path() / 'Scatterplots' / 'AllProcessPrediction' / 'legend.pgf', format='pgf')


conn = get_database_connection()

plot_scatter(connection=conn, instances=6000, save_legend=True)
plot_scatter(connection=conn, instances=6000, only_decision=True)
plot_scatter(connection=conn, instances=6000, flow_anomalies=0.3)
plot_scatter(connection=conn, instances=6000, flow_anomalies=0.3, only_decision=True)
plot_scatter(connection=conn, instances=6000, sensor_anomalies=0.3)
plot_scatter(connection=conn, instances=6000, sensor_anomalies=0.3, only_decision=True)
plot_scatter(connection=conn, instances=6000, sensor_anomalies=0.3, flow_anomalies=0.3)
plot_scatter(connection=conn, instances=6000, sensor_anomalies=0.3, flow_anomalies=0.3, only_decision=True)
plot_scatter(connection=conn, instances=10000)
plot_scatter(connection=conn, instances=10000, only_decision=True)
plot_scatter(connection=conn, instances=10000, sensor_anomalies=0.3, flow_anomalies=0.3)
plot_scatter(connection=conn, instances=10000, sensor_anomalies=0.3, flow_anomalies=0.3, only_decision=True)
conn.close()
