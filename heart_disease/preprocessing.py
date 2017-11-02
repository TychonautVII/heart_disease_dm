import numpy as np
from heart_disease.globals import code_path_str, data_path_str, outcome_to_learn
from sklearn.preprocessing import Imputer, StandardScaler
import pandas as pd
import logging


def load_raw_data(data_file_name_str):
    """This function loads files located in the global data_path_str directory
    matching the format of the processed.x.data files from the UCI repository.
    It returns a pandas data frame"""

    # Get Metadata
    meta_data_df = pd.read_csv(code_path_str+'meta_data.csv')
    # meta2pdtype = {'numeric':np.float64,'bool':np.bool,'categorical':'category'}
    meta2pdtype = {'numeric':np.float64,'bool':np.float64,'categorical':'category'}


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

def __clean_numeric(data_df, preprocessing_objc):

    # Impute Missing Values
    if 'numeric_imputer' not in preprocessing_objc:
        logging.info("Constructing New Numeric Imputer")
        numeric_imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
        numeric_imputer.fit(data_df)
        preprocessing_objc['numeric_imputer'] = numeric_imputer
    else:
        logging.info("Using Precomputed Numeric Imputer")
        numeric_imputer = preprocessing_objc['numeric_imputer']

    imputed_df = pd.DataFrame(numeric_imputer.transform(data_df), columns=data_df.columns)

    # Scale values to zero mean and unit variance
    if 'numeric_scaler' not in preprocessing_objc:
        logging.info("Constructing New Numeric Scalar")
        numeric_scaler = StandardScaler()
        numeric_scaler.fit(imputed_df)
        preprocessing_objc['numeric_scaler'] = numeric_scaler
    else:
        logging.info("Using Precomputed Numeric Scalar")
        numeric_scaler = preprocessing_objc['numeric_scaler']


    scaled_df = pd.DataFrame(numeric_scaler.transform(imputed_df), columns=imputed_df.columns)

    return scaled_df, preprocessing_objc

def __clean_bool(data_df, preprocessing_objc):

    if 'bool_imputer' not in preprocessing_objc:
        logging.info("Constructing New Boolean Imputer")
        bool_imputer = Imputer(missing_values=np.nan, strategy='most_frequent', axis=0, verbose=0, copy=True)
        bool_imputer.fit(data_df)
        preprocessing_objc['bool_imputer'] = bool_imputer
    else:
        logging.info("Using Precomputed Boolean Imputer")
        bool_imputer = preprocessing_objc['bool_imputer']

    imputed_df = pd.DataFrame(bool_imputer.transform(data_df),columns=data_df.columns)

    return imputed_df, preprocessing_objc

def __clean_categorical(data_df, preprocessing_objc):

    # Impute Missing Values
    if 'cat_imputer' not in preprocessing_objc:
        logging.info("Constructing New Categorical Imputer")
        cat_imputer = Imputer(missing_values=np.nan, strategy='most_frequent', axis=0, verbose=0, copy=True)
        cat_imputer.fit(data_df)
        preprocessing_objc['cat_imputer'] = cat_imputer
    else:
        logging.info("Using Precomputed Categorical Imputer")
        cat_imputer = preprocessing_objc['cat_imputer']

    imputed_df = pd.DataFrame(cat_imputer.transform(data_df), columns=data_df.columns, dtype=int)

    # One Hot Encode Categorical Features as Binary Features
    if 'one_hot_dict' not in preprocessing_objc:
        logging.info("Constructing New One Hot Dictionary")
        categories = list(imputed_df.columns)
        one_hot_dict = {}
        for cat in categories:
            one_hot_dict[cat] = np.unique(imputed_df[cat])
        preprocessing_objc['one_hot_dict'] = one_hot_dict
    else:
        logging.info("Using precomputed One Hot Dictionary")
        one_hot_dict = preprocessing_objc['one_hot_dict']
        # TODO: optionally Raise error / warning when value not in dictionary is in this dataframe

    one_hot_df = pd.DataFrame()

    for feature, values in one_hot_dict.items():
        for value in values:
            name = feature + '_is' + str(value)
            one_hot_df[name] = imputed_df[feature] == int(value)
            one_hot_df[name] = one_hot_df[name].astype(np.float)


    return one_hot_df, preprocessing_objc

def clean_data(data_df, preprocessing_objc={}):
    """You pass in a pandas dataframe read in load_raw_data, and it will impute missing values, encode enumerations, and convert all data to floats"""

    meta_data_df = pd.read_csv(code_path_str+'meta_data.csv')

    #----------------------------
    # Break input down by data type
    #----------------------------
    cols_in_frame = set(data_df.columns)

    bool_feats = set(meta_data_df[meta_data_df['datatype'] == 'bool']['name']);
    bool_feats = list(bool_feats.intersection(cols_in_frame))

    cat_feats = set(meta_data_df[meta_data_df['datatype'] == 'categorical']['name']);
    cat_feats = list(cat_feats.intersection(cols_in_frame))

    numeric_feats = set(meta_data_df[meta_data_df['datatype'] == 'numeric']['name']);
    numeric_feats = list(numeric_feats.intersection(cols_in_frame))

    # TODO: handle pass through columns. Anything not in Meta, but in Frame is a pass though.
    pass_through_cols = cols_in_frame - set(bool_feats) - set(cat_feats) - set(numeric_feats)

    #----------------------------
    # Clean Each datatype seperately
    #----------------------------
    numeric_df, preprocessing_objc = __clean_numeric(data_df.loc[:, numeric_feats],preprocessing_objc)
    cat_df, preprocessing_objc = __clean_categorical(data_df.loc[:, cat_feats], preprocessing_objc)
    bool_df, preprocessing_objc = __clean_bool(data_df.loc[:, bool_feats],preprocessing_objc)

    #----------------------------
    #  Combine Data Frames add in Truth Columnt
    #----------------------------
    out_df = numeric_df.merge(cat_df,how='outer',left_index=True,right_index=True)
    out_df = out_df.merge(bool_df,how='outer',left_index=True,right_index=True)

    for col in pass_through_cols:
        out_df[col] = data_df[col]

    return out_df, preprocessing_objc
