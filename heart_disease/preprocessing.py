import numpy as np
from heart_disease.globals import code_path_str, data_path_str

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import pandas as pd
import logging
logger = logging.getLogger(__name__)

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

class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        logger.debug("Calling OHE fit")

        total_feats = 0

        n_values_ = []
        feature_indices_ = []

        one_hot_dict = {}
        for col_idx in range(X.shape[1]):
            unique_elements = np.unique(X[:, col_idx])
            one_hot_dict[col_idx] = {'uniques':unique_elements,
                                     'out_start':total_feats}

            feature_indices_.append(total_feats)
            total_feats += len(unique_elements)
            n_values_.append(unique_elements)

        self.one_hot_dict_ = one_hot_dict
        self.total_feats_ = total_feats
        self.n_values_ = np.array(n_values_)
        self.feature_indices_ = np.array(feature_indices_)

        return self

    def transform(self, X):
        """ Returns the transformed x after one hot encoding
            X : array-like, shape = (n_samples, n_features)
        """

        logger.debug("Calling OHE transform")

        X_out = np.zeros((X.shape[0],self.total_feats_))

        out_idx = 0
        for column, this_dict in self.one_hot_dict_.items():
            out_idx = this_dict['out_start']
            for value in this_dict['uniques']:
                bool_array = X[:,column] == int(value)
                X_out[:,out_idx] = bool_array.astype(np.float)
                out_idx += 1

        return X_out

def __clean_numeric(data_df, preprocessing_objc):

    # Impute Missing Values
    if 'numeric_imputer' not in preprocessing_objc:
        logger.debug("Constructing New Numeric Imputer")
        numeric_imputer = Imputer( strategy='mean',  verbose=0)
        numeric_imputer.fit(data_df)
        preprocessing_objc['numeric_imputer'] = numeric_imputer
    else:
        logger.debug("Using Precomputed Numeric Imputer")
        numeric_imputer = preprocessing_objc['numeric_imputer']

    imputed_df = pd.DataFrame(numeric_imputer.transform(data_df), columns=data_df.columns)

    # Scale values to zero mean and unit variance
    if 'numeric_scaler' not in preprocessing_objc:
        logger.debug("Constructing New Numeric Scalar")
        numeric_scaler = StandardScaler()
        numeric_scaler.fit(imputed_df)
        preprocessing_objc['numeric_scaler'] = numeric_scaler
    else:
        logger.debug("Using Precomputed Numeric Scalar")
        numeric_scaler = preprocessing_objc['numeric_scaler']


    scaled_df = pd.DataFrame(numeric_scaler.transform(imputed_df), columns=imputed_df.columns)

    return scaled_df, preprocessing_objc

def __clean_bool(data_df, preprocessing_objc):

    if 'bool_imputer' not in preprocessing_objc:
        logger.debug("Constructing New Boolean Imputer")
        bool_imputer = Imputer( strategy='most_frequent',  verbose=0)
        bool_imputer.fit(data_df)
        preprocessing_objc['bool_imputer'] = bool_imputer
    else:
        logger.debug("Using Precomputed Boolean Imputer")
        bool_imputer = preprocessing_objc['bool_imputer']

    imputed_df = pd.DataFrame(bool_imputer.transform(data_df),columns=data_df.columns)

    return imputed_df, preprocessing_objc

def __clean_categorical(data_df, preprocessing_objc):

    # Impute Missing Values
    if 'cat_imputer' not in preprocessing_objc:
        logger.debug("Constructing New Categorical Imputer")
        cat_imputer = Imputer( strategy='most_frequent',  verbose=0)
        cat_imputer.fit(data_df)
        preprocessing_objc['cat_imputer'] = cat_imputer
    else:
        logger.debug("Using Precomputed Categorical Imputer")
        cat_imputer = preprocessing_objc['cat_imputer']

    imputed_df = pd.DataFrame(cat_imputer.transform(data_df), columns=data_df.columns, dtype=int)

    # One Hot Encode Categorical Features as Binary Features
    if 'one_hot_dict' not in preprocessing_objc:
        logger.debug("Constructing New One Hot Dictionary")
        categories = list(imputed_df.columns)
        one_hot_dict = {}
        for cat in categories:
            one_hot_dict[cat] = np.unique(imputed_df[cat])
        preprocessing_objc['one_hot_dict'] = one_hot_dict
    else:
        logger.debug("Using precomputed One Hot Dictionary")
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

    meta_data_df = pd.read_csv(code_path_str + 'meta_data.csv')

    # ----------------------------
    # Break input down by data type
    # ----------------------------
    cols_in_frame = set(data_df.columns)

    bool_feats = set(meta_data_df[meta_data_df['datatype'] == 'bool']['name'])
    bool_feats = list(bool_feats.intersection(cols_in_frame))

    cat_feats = set(meta_data_df[meta_data_df['datatype'] == 'categorical']['name'])
    cat_feats = list(cat_feats.intersection(cols_in_frame))

    numeric_feats = set(meta_data_df[meta_data_df['datatype'] == 'numeric']['name'])
    numeric_feats = list(numeric_feats.intersection(cols_in_frame))

    pass_through_cols = cols_in_frame - set(bool_feats) - set(cat_feats) - set(numeric_feats)

    # ----------------------------
    # Clean Each datatype seperately
    # ----------------------------
    numeric_df, preprocessing_objc = __clean_numeric(data_df.loc[:, numeric_feats], preprocessing_objc)
    cat_df, preprocessing_objc = __clean_categorical(data_df.loc[:, cat_feats], preprocessing_objc)
    bool_df, preprocessing_objc = __clean_bool(data_df.loc[:, bool_feats], preprocessing_objc)


    # # START HACK TO TEST ONE HOT ENCODER TODO: Remove
    #
    #
    # ohe = OneHotEncoder()
    # matrix = np.array(data_df.loc[:, cat_feats])
    #
    # cat_imputer = Imputer( strategy='most_frequent',  verbose=0)
    # cat_imputer.fit(matrix)
    #
    # matrix = cat_imputer.transform(matrix)
    #
    # ohe.fit(matrix)
    # out = ohe.transform(matrix)
    #
    # # END HACK


    # ----------------------------
    #  Combine Data Frames add in Truth Columnt
    # ----------------------------
    out_df = numeric_df.merge(bool_df, how='outer', left_index=True, right_index=True)
    out_df = out_df.merge(cat_df, how='outer', left_index=True, right_index=True)

    for col in pass_through_cols:
        out_df[col] = data_df[col]

    return out_df, preprocessing_objc


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, path2meta=code_path_str+'meta_data.csv',
                       column_names_list=None,
                       transformers_by_type={'numeric':[Imputer(strategy='mean',  verbose=0),
                                                        StandardScaler()],
                                               'bool':[Imputer(strategy='most_frequent',  verbose=0)],
                                               'categorical':[Imputer(strategy='most_frequent',  verbose=0),
                                                              OneHotEncoder()]}):

        # TODO Have some Standards across all transformers, but make anything changeable kv

        # https: // stats.stackexchange.com / questions / 177082 / sklearn - combine - multiple - feature - sets - in -pipeline
        # ItemSelector
        # http: // scikit - learn.org / stable / auto_examples / hetero_feature_union.html
        # Path to the Feature Metadata
        # http: // scikit - learn.org / stable / developers / contributing.html
        # http: // scikit - learn.org / stable / modules / generated / sklearn.model_selection.GridSearchCV.html


        # http: // scikit - learn.org / stable / modules / pipeline.html  # pipeline
        self.path2meta = path2meta # Path to the Feature Metadata





        # Column Names of the Matrix X
        if column_names_list is None:
            # prevents silliness with mutable default input arguements
            self.column_names_list = []
        else:
            self.column_names_list = column_names_list

        # Indexes by Datatype
        self.index_by_type = {}

        #
        self.transformers_by_type = transformers_by_type


        # Get Feature Indexes, then Get Submatrix, Then Apply transform

    def _get_feature_indexes(self, this_type, meta_data_df):
        column_name_list = self.column_names_list

        feat_names = set(meta_data_df[meta_data_df['datatype'] == this_type]['name'])
        feat_names = list(feat_names.intersection(set(column_name_list)))

        feat_indexs = list()

        for feat in feat_names:
            feat_indexs.append(column_name_list.index(feat))

        self.index_by_type[this_type] = feat_indexs

    def _get_submatrix(self, this_type, X):
        feat_indexs = self.index_by_type[this_type]

        sub_X = np.zeros((X.shape[0], len(feat_indexs)))

        for out_idx, in_idx in enumerate(feat_indexs):
            sub_X[:, out_idx] = X[:, in_idx]

        return sub_X

    def _do_fits(self, this_type,X):
        transformers = self.transformers_by_type[this_type]

        for transformer in transformers:
            logger.debug(transformer)
            X = transformer.fit_transform(X)

        return X

    def _do_transforms(self, this_type, X):
        transformers = self.transformers_by_type[this_type]

        for transformer in transformers:
             X = transformer.transform(X)

    def fit(self, X, y=None):
        """ Returns the transformed x after one hot encoding
            X : array-like, shape = (n_samples, n_features)
        """

        logger.debug("Calling fit method")

        meta_data_df = pd.read_csv(self.path2meta)

        total = None
        for type in ['numeric','bool','categorical']:
            self._get_feature_indexes(type, meta_data_df)
            X_this_type = self._get_submatrix(type,X)

            if total is None:
                total = self._do_fits(type,X_this_type)
            else:
                total = np.hstack( (total, self._do_fits(type,X_this_type)) )

        return self

    def transform(self, X):
        """ Returns the transformed x after one hot encoding
            X : array-like, shape = (n_samples, n_features)
        """
        logger.debug("Calling transform method")

        def _do_fits(self, this_type, X)

        return