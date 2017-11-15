import numpy as np
import logging
import pandas as pd
from copy import deepcopy as cp
import matplotlib
matplotlib.use('agg')
matplotlib = matplotlib.reload(matplotlib)

import heart_disease.preprocessing as hdpp
import heart_disease.parameter_optimization as hdpo
from heart_disease.globals import data_path, output_path


# Create Loger Object
LOG_FMT = '%(asctime)s|%(name)s|%(levelname)s|%(message)s'

handlers = [logging.FileHandler('grid_search.log'), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format=LOG_FMT,handlers=handlers)
logger = logging.getLogger(__name__)

# Get Metadata


if __name__ =='__main__':
    data_file_name_str = 'train_validation.cleveland.csv'
    data_df = hdpp.load_data(data_path + data_file_name_str)
    hdpo.execute_grid_search(data_df,data_df['ispos_truth'])
    # TODO: add PCA to cleaner, make it an obvious paramater to be passed into cleaner, No Wrap PCA and put it in there!
