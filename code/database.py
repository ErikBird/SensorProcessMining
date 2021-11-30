import sqlite3
from datetime import datetime
from sqlite3 import Error
import progressbar

from utils import get_database_connection


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def initialize_table(conn):
    sql_create_dataset_table = """ CREATE TABLE IF NOT EXISTS dataset (
                                        id integer PRIMARY KEY,
                                        process text NOT NULL,
                                        creation_date text NOT NULL,
                                        only_decision boolean NOT NULL,
                                        num_instances integer NOT NULL,
                                        max_sequence_length integer NOT NULL,
                                        file_path text,
                                        anomalies_flow text,
                                        anomalies_sensor text,
                                        num_events integer,
                                        generation_setting text
                                    ); """

    sql_create_experiment_table = """ CREATE TABLE IF NOT EXISTS experiment (
                                    id integer PRIMARY KEY,
                                    preprocessing text,
                                    train_dataset integer,
                                    test_dataset integer,
                                    filename text,
                                    training_duration text,
                                    hyperparameter text,
                                    epochs integer,
                                    accuracy float,
                                    kappa float,
                                    precision float,
                                    recall float,
                                    AUC float,
                                    f1 float,
                                    FOREIGN KEY (train_dataset) REFERENCES dataset (id),
                                    FOREIGN KEY (test_dataset) REFERENCES dataset (id)
                                );"""

    sql_create_anomaly_table = """ CREATE TABLE IF NOT EXISTS anomaly (
                                    id integer PRIMARY KEY,
                                    preprocessing text,
                                    train_sensor_anomalies float,
                                    train_flow_anomalies float,
                                    eval_sensor_anomalies float,
                                    eval_flow_anomalies float,
                                    anomalies text,
                                    graph text,
                                    accuracy float,
                                    kappa float,
                                    precision float,
                                    recall float,
                                    f1 float,
                                    threshold float
                                );"""
    # create tables
    if conn is not None:
        create_table(conn, sql_create_dataset_table)
        create_table(conn, sql_create_experiment_table)
        create_table(conn, sql_create_anomaly_table)
    else:
        print("Error! cannot create the database connection.")


def create_dataset(conn, process, generation_setting, max_sequence_length, num_instances, file_path,
                   creation_date=str(datetime.now().time()),
                   only_decision=False,
                   anomalies_flow='0.0',
                   anomalies_sensor='0.0',
                   num_events=0):
    sql = ''' INSERT INTO dataset(process,creation_date,only_decision,num_instances,file_path, anomalies_flow, anomalies_sensor, num_events, generation_setting, max_sequence_length)
              VALUES(?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (
        process, creation_date, only_decision, num_instances, file_path, anomalies_flow, anomalies_sensor, num_events,
        generation_setting, max_sequence_length))
    conn.commit()
    return cur.lastrowid


def create_experiment(conn, preprocessing, train_dataset, test_dataset, filename, training_duration, hyperparameter,
                      epochs, accuracy, kappa, precision, recall, AUC=0, f1=.0):
    sql = ''' INSERT INTO experiment(preprocessing,train_dataset,test_dataset,filename,training_duration,hyperparameter,
    epochs,accuracy, kappa,precision,recall,AUC,f1)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (preprocessing, train_dataset, test_dataset, filename, training_duration, hyperparameter,
                      epochs, accuracy, kappa, precision, recall, AUC, f1))
    conn.commit()
    return cur.lastrowid


def create_anomaly_experiment(conn, preprocessing, train_sensor_anomalies, train_flow_anomalies, eval_sensor_anomalies,
                              eval_flow_anomalies
                              , anomalies, graph, accuracy, kappa, precision, recall, f1, threshold=None):
    sql = ''' INSERT INTO anomaly(preprocessing,
    train_sensor_anomalies,train_flow_anomalies,eval_sensor_anomalies,eval_flow_anomalies
    ,anomalies,graph,accuracy, kappa,precision,recall,f1,threshold)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (preprocessing,
                      train_sensor_anomalies, train_flow_anomalies, eval_sensor_anomalies, eval_flow_anomalies
                      , anomalies, graph, accuracy, kappa, precision, recall, f1, threshold))
    conn.commit()
    return cur.lastrowid
