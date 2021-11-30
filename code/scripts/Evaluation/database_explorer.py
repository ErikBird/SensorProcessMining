#%%from ProcessSimulator.src.simulator.process_model import DatasetStage
from ProcessSimulator.src.simulator.process_model import DatasetStage
from utils import get_database_connection, get_assets_path, get_matplotlib_size, get_color_hex_list
import matplotlib.pyplot as plt
import pandas as pd


conn = get_database_connection()
'''
df = pd.read_sql_query("""SELECT
                                experiment.id,
                                experiment.preprocessing,
                                train_dataset.num_instances,
                                train_dataset.process,
                                train_dataset.anomalies_sensor,
                                train_dataset.anomalies_flow,
                                kappa
                            FROM
                                experiment
                            LEFT JOIN dataset as test_dataset ON experiment.test_dataset = test_dataset.id
                            LEFT JOIN dataset as train_dataset ON experiment.train_dataset = train_dataset.id
                            WHERE test_dataset.generation_setting=:stage AND train_dataset.num_instances=6000;""",
                       conn,
                       params={"preprocessing": 'baseline_id', "stage": DatasetStage.TEST.value})
'''
df = pd.read_sql_query("""SELECT
                                    kappa,
                                    f1,
                                    precision,
                                    recall,
                                    preprocessing,
                                    graph
                                FROM
                                    anomaly
                               WHERE preprocessing ='kal_norm_cnn';""",
                       conn,
                       params={"preprocessing": 'baseline_id', "stage": DatasetStage.TEST.value})




pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None, "display.max_columns", None)


print(df)
