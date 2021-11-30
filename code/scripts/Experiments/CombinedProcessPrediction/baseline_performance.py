from ProcessSimulator.src.processes import LaboratorySimulator
from ProcessSimulator.src.processes.hospital_process.hospital_process import HospitalSimulator
from ProcessSimulator.src.simulator.process_model import DatasetStage
from utils import get_database_connection, get_assets_path, get_matplotlib_size, get_color_hex_list
import matplotlib.pyplot as plt
import pandas as pd


conn = get_database_connection()
print('Baseline ID')
print('Test Dataset')
df = pd.read_sql_query("""SELECT
                                train_dataset.num_instances,
                                test_dataset.process,
                                accuracy,
                                kappa,
                                precision,
                                recall,
                                f1
                            FROM
                                experiment
                            LEFT JOIN dataset as test_dataset ON experiment.test_dataset = test_dataset.id
                            LEFT JOIN dataset as train_dataset ON experiment.train_dataset = train_dataset.id
                            WHERE experiment.preprocessing=:preprocessing AND test_dataset.generation_setting=:stage ;""",
                       conn,
                       params={"preprocessing": 'baseline_id', "stage": DatasetStage.TEST.value})

print(df.to_latex(index=False))

print('Test Only Decision Dataset')
df = pd.read_sql_query("""SELECT
                                train_dataset.num_instances,
                                test_dataset.process,
                                accuracy,
                                kappa,
                                precision,
                                recall,
                                f1
                            FROM
                                experiment
                            LEFT JOIN dataset as test_dataset ON experiment.test_dataset = test_dataset.id
                            LEFT JOIN dataset as train_dataset ON experiment.train_dataset = train_dataset.id
                            WHERE experiment.preprocessing=:preprocessing AND test_dataset.generation_setting=:stage ;""",
                       conn,
                       params={"preprocessing": 'baseline_id', "stage": DatasetStage.TEST_DECISION.value,
                               "num_instances": 1100})

print(df.to_latex(index=False))

print('Baseline Attribute')
print('Test Dataset')
df = pd.read_sql_query("""SELECT
                                train_dataset.num_instances,
                                test_dataset.process,
                                accuracy,
                                kappa,
                                precision,
                                recall,
                                f1
                            FROM
                                experiment
                            LEFT JOIN dataset as test_dataset ON experiment.test_dataset = test_dataset.id
                            LEFT JOIN dataset as train_dataset ON experiment.train_dataset = train_dataset.id
                            WHERE experiment.preprocessing=:preprocessing AND test_dataset.generation_setting=:stage ;""",
                       conn,
                       params={"preprocessing": 'baseline_attr', "stage": DatasetStage.TEST.value,
                               "num_instances": 1100})

print(df.to_latex(index=False))

print('Test Only Decision Dataset')
df = pd.read_sql_query("""SELECT
                                train_dataset.num_instances,
                                test_dataset.process,
                                accuracy,
                                kappa,
                                precision,
                                recall,
                                f1
                            FROM
                                experiment
                            LEFT JOIN dataset as test_dataset ON experiment.test_dataset = test_dataset.id
                            LEFT JOIN dataset as train_dataset ON experiment.train_dataset = train_dataset.id
                            WHERE experiment.preprocessing=:preprocessing AND test_dataset.generation_setting=:stage ;""",
                       conn,
                       params={"preprocessing": 'baseline_attr', "stage": DatasetStage.TEST_DECISION.value,
                               "num_instances": 1100})

print(df.to_latex(index=False))
