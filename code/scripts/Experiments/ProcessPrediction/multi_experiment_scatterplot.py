from ProcessSimulator.src.processes import LaboratorySimulator
from ProcessSimulator.src.processes.hospital_process.hospital_process import HospitalSimulator
from ProcessSimulator.src.simulator.process_model import DatasetStage
from utils import get_database_connection, get_assets_path, get_matplotlib_size, get_color_hex_list
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import matplotlib
import progressbar
import numpy as np
import matplotlib.colors


def get_dataframe(connection, instances, flow_anomalies=0.0, sensor_anomalies=0.0):
    laboratory_process = LaboratorySimulator()
    hospital_process = HospitalSimulator()
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
                                      params={"name": laboratory_process.__name__, "stage": DatasetStage.TEST.value,
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
                                    params={"name": hospital_process.__name__, "stage": DatasetStage.TEST.value,
                                            "num_instances": instances, 'flow_anomalies': flow_anomalies,
                                            'sensor_anomalies': sensor_anomalies})

    df = pd.merge(laboratory_df, hospital_df, on='preprocessing', how='left')
    return df


conn = get_database_connection()
df_no_anom = get_dataframe(connection=conn, instances=6000, flow_anomalies=0.0, sensor_anomalies=0.0)
df_flow = get_dataframe(connection=conn, instances=6000, flow_anomalies=0.3, sensor_anomalies=0.0)
df_sensor = get_dataframe(connection=conn, instances=6000, flow_anomalies=0.0, sensor_anomalies=0.3)
df_no_both = get_dataframe(connection=conn, instances=6000, flow_anomalies=0.3, sensor_anomalies=0.3)
fig, ax = plt.subplots(1, 1, figsize=get_matplotlib_size(418.25555))  # 1, 1, figsize=get_matplotlib_size(418.25555))
colors = get_color_hex_list()

models = ['norm_rnn', 'norm_cnn', 'baseline_id', 'baseline_attr', 'baseline_sensor']
for df, marker, label in [(df_no_anom, 'o', 'no anomalies'), (df_flow, 'x', 'flow anomalies'),
                          (df_sensor, '+', 'sensor anomalies'), (df_no_both, '*', 'both anomalies')]:
    filtered_df = df[df['preprocessing'].isin(models)]
    for index, row in filtered_df.iterrows():
        ax.scatter([row['kappa_x']], [row['kappa_y']], label=row['preprocessing'], alpha=1.0,
               marker=marker)
ax.legend(bbox_to_anchor=(1.05, 1), prop={'size': 6})
fig.tight_layout()
#name = experiments + decision + flow_name + str(instances) + ".pgf"
#fig.savefig(get_assets_path() / 'Experiments' / 'Scatterplots' / name, format='pgf')
plt.show()
conn.close()
