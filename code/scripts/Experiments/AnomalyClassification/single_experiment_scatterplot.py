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


def plot_scatter(connection, instances, only_first=False):
    if only_first:
        anomalies = 'only_first'
    else:
        anomalies = 'all'
    laboratory_process = LaboratorySimulator()
    hospital_process = HospitalSimulator()

    laboratory_df = pd.read_sql_query("""SELECT
                                    f1,
                                    preprocessing
                                FROM
                                    anomaly
                               WHERE anomalies =:anomalies
                               AND graph =:graph_name;""",
                                      connection,
                                      params={"graph_name": str(instances) + laboratory_process.__name__,
                                              'anomalies': anomalies})

    hospital_df = pd.read_sql_query("""SELECT
                                    f1,
                                    preprocessing
                                FROM
                                    anomaly
                               WHERE anomalies =:anomalies
                               AND graph =:graph_name;""",
                                    connection,
                                    params={"graph_name": str(instances) + hospital_process.__name__,
                                            'anomalies': anomalies})

    df = pd.merge(laboratory_df, hospital_df, on='preprocessing', how='left')
    df.sort_values(by=['preprocessing'], inplace=True, ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))  # 1, 1, figsize=get_matplotlib_size(418.25555))
    for index, row in df.iterrows():
        marker = "x"
        if 'baseline' in row['preprocessing']:
            marker = 'o'
        ax.scatter([row['f1_x']], [row['f1_y']], label=row['preprocessing'], alpha=1.0,
                   marker=marker)

    ax.set_xlabel(r'$F_1$ Laboratory Dataset')
    ax.set_ylabel(r'$F_1$  Hospital Dataset')

    #ax.legend(bbox_to_anchor=(1.05, 1), prop={'size': 6})

    #name = str(instances) + decision + flow_name + ".pgf"

    #ax.set_title(name)
    ax.set_ylim([0, 0.5])
    ax.set_xlim([0, 0.5])
    fig.tight_layout()
    name = str(instances)+'_'+anomalies+'.pgf'
    fig.savefig(get_assets_path() / 'Scatterplots' / 'ClassificationComparison' / name, format='pgf')
    plt.show()


    figsize = (2, 3)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig(get_assets_path() / 'Scatterplots' / 'ClassificationComparison' / 'legend.pgf', format='pgf')


conn = get_database_connection()
plot_scatter(connection=conn, instances=10000, only_first=True)
plot_scatter(connection=conn, instances=6000, only_first=True)
plot_scatter(connection=conn, instances=10000, only_first=False)
plot_scatter(connection=conn, instances=6000, only_first=False)
conn.close()
