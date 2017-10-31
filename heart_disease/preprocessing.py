import numpy as np
from heart_disease.globals import code_path_str, data_path_str
import pandas as pd


def load_raw_data(data_file_name_str):
    """This function loads files located in the global data_path_str directory
    matching the format of the processed.x.data files from the UCI repository.
    It returns a pandas data frame"""

    # Get Metadata
    meta_data_df = pd.read_csv(code_path_str+'meta_data.csv')
    meta2pdtype = {'numeric':np.float64,'bool':np.bool,'categorical':'category'}


    # Load in the actual data
    column_names = ['age', 'sex', 'cp', 'trestbps',
                    'chol', 'fbs', 'restecg', 'thalach',
                    'exang', 'oldpeak', 'slope',
                    'ca', 'thal', 'num']

    # Associate Columns with datatypes
    dtype_map = {}
    for col in column_names:
        meta_datatype = meta_data_df['datatype'].loc[meta_data_df['name'] == col].iloc[0]
        dtype_map[col] = meta2pdtype[meta_datatype]

    # Read in the data
    data_df = pd.read_csv((data_path_str + data_file_name_str),
                          names=column_names, na_values='?',dtype=dtype_map)

    # Basic Pre-processing:
    data_df['ispos_truth'] = data_df['num'] != '0'
    data_df.drop('num', 1, inplace=True)

    return data_df

def clean_data(df, preprocessing_objc=None):
    pass
