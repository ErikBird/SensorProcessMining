from ProcessSimulator.src.processes.hospital_process.hospital_process import HospitalSimulator
from utils import get_database_connection
from dataset import DataConvolut

conn = get_database_connection()
graph = HospitalSimulator()
data_convolut = DataConvolut(connection=conn, instances_train=10, instances_dev=10, instances_test=10,
                             graph=graph,
                             create_only_decisions=True,
                             batch_size=128,
                             anomalies_sensor=1.0,
                             anomalies_flow=1.0
                             )
data_convolut.visualize_flow(flow_description=data_convolut.anomalies_flow_description,
                             sensor_description=data_convolut.anomalies_sensor_description)
