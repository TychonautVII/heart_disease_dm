import numpy as np
import logging
import pandas as pd
from copy import deepcopy as cp
import matplotlib
matplotlib.use('agg')
matplotlib = matplotlib.reload(matplotlib)


from heart_disease.globals import code_path_str, data_path_str
import heart_disease.preprocessing as pre_proc


# Create Loger Object
LOG_FMT = '%(asctime)s|%(name)s|%(levelname)s|%(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
logger = logging.getLogger(__name__)

# import logging
# logger = logging.getLogger('heart_disease')
#
# sh = logging.StreamHandler()
# formatter = logging.Formatter(LOG_FMT)
# sh.setFormatter(formatter)
# logger.addHandler(sh)


# Get Metadata
meta_data_df = pd.read_csv(code_path_str+'meta_data.csv')

data_file_name_str = 'processed.va.data.txt'

data_df = pre_proc.load_raw_data(data_file_name_str)

clean_df, preproc_dict = pre_proc.clean_data(data_df)
clean_df.head()

# New Method


if __name__ =='__main__':
    print("is Main")