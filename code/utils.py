import os
import sqlite3
from pathlib import Path
from sqlite3 import Error
from typing import List, Tuple

from dask import dataframe as dd
from PIL import Image


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_root_dataset_drive() -> Path:
    # Make sure to set a path or use project_root:
    return get_project_root()


def get_data_path() -> Path:
    return get_project_root() / 'code' / 'data'


def get_bosch_data_path() -> Path:
    return get_data_path() / 'bosch'


def get_patient_flow_path() -> Path:
    return get_project_root() / 'code' / 'ProcessSimulator' / 'src' / 'processes' / 'hospital_process'


def get_abp_train_file_path() -> Path:
    return get_patient_flow_path() / 'ABP' / 'data_train.pkl'


def get_abp_dev_file_path() -> Path:
    return get_patient_flow_path() / 'ABP' / 'data_dev.pkl'


def get_abp_test_file_path() -> Path:
    return get_patient_flow_path() / 'ABP' / 'data_test.pkl'


def get_ecg_train_normal_file_path() -> Path:
    return get_patient_flow_path() / 'ECG' / 'ecg_normal_train.pkl'


def get_ecg_train_anormal_file_path() -> Path:
    return get_patient_flow_path() / 'ECG' / 'ecg_anomal_train.pkl'


def get_ecg_dev_normal_file_path() -> Path:
    return get_patient_flow_path() / 'ECG' / 'ecg_normal_dev.pkl'


def get_ecg_dev_anormal_file_path() -> Path:
    return get_patient_flow_path() / 'ECG' / 'ecg_anomal_dev.pkl'


def get_ecg_test_normal_file_path() -> Path:
    return get_patient_flow_path() / 'ECG' / 'ecg_normal_test.pkl'


def get_ecg_test_anormal_file_path() -> Path:
    return get_patient_flow_path() / 'ECG' / 'ecg_anomal_test.pkl'


def get_dataset_output_path() -> Path:
    return get_root_dataset_drive() / '.out' / 'datasets'


def get_graphviz_cache_path() -> Path:
    return get_project_root() / '.graphviz_cache'


def get_datasequence_output_path() -> Path:
    return get_project_root() / '.out' / 'datasequence'


def get_log_output_path() -> Path:
    return get_project_root() / '.out' / 'logs'


def get_model_output_path() -> Path:
    return get_project_root() / '.out' / 'models'

def get_animation_output_path() -> Path:
    return get_project_root() / '.out' / 'animations'


def get_output_path() -> Path:
    return get_project_root() / '.out'


def get_assets_path() -> Path:
    return get_project_root() / 'tex' / 'latex' / 'tuda-ci' / 'assets'


def get_database_connection():
    """ create a database connection to a SQLite database """
    db_file = get_output_path() / 'experiments.db'
    conn = None
    try:
        print('try to connect to %s' % db_file)
        conn = sqlite3.connect(db_file)
        # Set journal mode to WAL.
        # conn.execute('pragma journal_mode=wal')
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn


def get_matplotlib_size(width_pt, fraction=1, subplots=(1, 1)):
    """Get figure dimensions of matplotlib PGF to fit nicely in our latex document.

    Latex command to get the width_pt: \the\textwidth

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


class GifReader:
    pil_image = None
    length = -1

    def __init__(self, file_path):
        self._open(file_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _open(self, file_path):
        if self.pil_image != None:
            raise ValueError(f'File is already open')

        if os.path.isfile(file_path) == False:
            raise ValueError(f'File {file_path} doesn\'t exist')

        with open(file_path, 'rb') as f:
            if f.read(6) not in (b'GIF87a', b'GIF89a'):
                raise ValueError('Not a valid GIF file')

        self.pil_image = Image.open(file_path)
        self.length = self.pil_image.n_frames

    def close(self):
        if self.pil_image == None: return
        self.pil_image.close()
        self.pil_image = None

    def get_length(self):
        if self.pil_image == None:
            raise ValueError('File is closed')
        return self.length

    def seek(self, n):
        if self.pil_image == None:
            raise ValueError('File is closed')
        if n >= self.length:
            raise ValueError(f'Seeking frame {n} while maximum is {self.length-1}')

        self.pil_image.seek(n)
        a = self.pil_image.convert ('RGB')
        return a


def get_color_hex_list() -> List:
    """
    Return Hex codes of TU Darmstadt CI in Color category "c"
    The order is determined to be distinguishable

    :return: List of Hex codes
    """
    return ['#B90F22', '#004E8A', '#D7AC00', '#951169', '#00689D', '#611C73', '#D28700', '#008877', '#CC4C03',
            '#B1BD00', '#CC4C03', '#7FAB16']


def get_annotated_data() -> Tuple:
    train_categorical_df = dd.read_csv(get_bosch_data_path() / 'train_categorical_repaired_replaced.csv').set_index(
        'Id')
    train_numeric_df = dd.read_csv(get_bosch_data_path() / 'train_numeric_replaced.csv').set_index('Id')
    train_date_df = dd.read_csv(get_bosch_data_path() / 'train_date.csv').set_index('Id')
    return train_categorical_df, train_numeric_df, train_date_df


def get_dataframe_value_date():
    train_categorical_df, train_numeric_df, train_date_df = get_annotated_data()
    # df = train_categorical_df.append(train_numeric_df)
    # df = df.append(train_date_df)
    df_value = dd.merge(train_categorical_df, train_numeric_df, left_on='Id',
                        right_on='Id')  # , left_index=True, right_index=True)#
    df = dd.merge(df_value, train_date_df, left_on='Id', right_on='Id')
    # df = dd.merge(df, train_date_df, left_index=True, right_on='Id')
    return df  # , train_date_df
