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


def plot_scatter(connection, graph, instances, only_first=False, only_decision=False):
    if only_first:
        anomalies = 'only_first_grid'
    else:
        anomalies = 'all_grid'
    if only_decision:
        anomalies += '_decision'

    df = pd.read_sql_query("""SELECT
                                    f1,
                                    threshold,
                                    preprocessing
                                FROM
                                    anomaly
                               WHERE anomalies =:anomalies
                               AND graph =:graph_name;""",
                           connection,
                           params={"graph_name": str(instances) + graph.__name__,
                                   'anomalies': anomalies})


    fig, ax = plt.subplots(1, 1, figsize=(4,3))  # 1, 1, figsize=get_matplotlib_size(418.25555))

    for id in df['preprocessing'].unique():
        q_df = df.query("preprocessing == '%s'"%id)
        if 'baseline' in id:
            marker ='.-'
        else:
            marker ='*-'
        ax.plot(q_df['threshold'], q_df['f1'], marker, label=id)

    ax.set_xlabel(r'Discrimination Threshold $\tau$')
    ax.set_ylabel(r'$F_1$')

    #ax.legend()

    #ax.set_ylim([0, 0.5])
    # ax.set_xlim([0, 0.5])
    fig.tight_layout()
    name = str(instances) + graph.__name__+'_'+anomalies + '.pgf'
    #plt.title(name)
    fig.savefig(get_assets_path() / 'Scatterplots' / 'ClassificationGrid' / name, format='pgf')
    plt.show()

    figsize = (2, 3)
    fig_leg = plt.figure(figsize=figsize)
    ax_leg = fig_leg.add_subplot(111)
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig(get_assets_path() / 'Scatterplots' / 'ClassificationGrid' / 'legend.pgf', format='pgf')

laboratory_process = LaboratorySimulator()
hospital_process = HospitalSimulator()
conn = get_database_connection()
#plot_scatter(connection=conn, graph=laboratory_process, instances=10000, only_first=True)
#plot_scatter(connection=conn, graph=hospital_process, instances=10000, only_first=True)
plot_scatter(connection=conn, graph=laboratory_process, instances=10000, only_first=False)
plot_scatter(connection=conn, graph=hospital_process, instances=10000, only_first=False)

conn.close()
