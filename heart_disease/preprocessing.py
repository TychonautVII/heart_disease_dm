import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import logging
logger = logging.getLogger(__name__)

def load_raw_data(data_path, meta_path):
    """This function loads files located in the global data_path_str directory
    matching the format of the processed.x.data files from the UCI repository.
    It returns a pandas data frame"""

    # Get Metadata
    meta_data_df = pd.read_csv(meta_path)
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
    data_df = pd.read_csv((data_path),
                          names=column_names, na_values='?',dtype=dtype_map)

    # Basic Pre-processing:
    data_df['ispos_truth'] = data_df['num'] != '0'
    data_df.drop('num', 1, inplace=True)

    return data_df

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """A customized one-hot encoding method, as sklearn made an assumption about the enumerations not true in my dataset"""
    def __init__(self):
        self.is_fitted = False

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
        self.is_fitted=True

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

    def get_new_column_names(self, old_col_name=None):
        """Can only be called after fit, returns the new columns names"""
        if old_col_name == None:
            old_col_name = list(range(len(self.one_hot_dict_)))
            old_col_name = [str(name) for name in old_col_name]

        new_names = list()

        for feat_idx, ohd in self.one_hot_dict_.items():
            for value in ohd['uniques']:
                name = old_col_name[feat_idx] + '_is' + str(int(value))
                new_names.append(name)

        return new_names

class ItemSelector(BaseEstimator,TransformerMixin):
    """Downselects a subset of features for processing,
       following patterned at http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html"""
    def __init__(self, key_list=[]):
        self.key_list = key_list

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        return np.array(dataframe.loc[:,self.key_list])

class DataCleaner(object):
    def __init__(self, path2meta, verbose=0):

        meta_data_df = pd.read_csv(path2meta)

        self.numeric_feat_list = list(meta_data_df[meta_data_df['datatype'] == 'numeric']['name'])
        self.bool_feat_list = list(meta_data_df[meta_data_df['datatype'] == 'bool']['name'])
        self.categorical_feat_list = list(meta_data_df[meta_data_df['datatype'] == 'categorical']['name'])

        self.CleaningPipeline = FeatureUnion(
            transformer_list=[
                ('numeric_pipe', Pipeline([('selector', ItemSelector(key_list=self.numeric_feat_list)),
                                           ('Imputer',  Imputer(strategy='mean', verbose=verbose)),
                                           ('scalar',   StandardScaler())])),
                ('bool_pipe', Pipeline([('selector', ItemSelector(key_list=self.bool_feat_list)),
                                        ('Imputer',  Imputer(strategy='most_frequent',  verbose=verbose)),
                                        ('scalar', StandardScaler())])),
                ('cat_pipe', Pipeline([('selector', ItemSelector(key_list=self.categorical_feat_list)),
                                       ('Imputer',  Imputer(strategy='most_frequent',  verbose=verbose)),
                                       ('encoder',  OneHotEncoder()),
                                       ('scalar', StandardScaler()) ])),

                ]
        )

    def get_output_col_neams(self):
        "Get the column names in order of the features post cleaning"
        ohe = self.CleaningPipeline.get_params()['cat_pipe__encoder']

        if ohe.is_fitted:
            new_cat_feat_list = ohe.get_new_column_names(self.categorical_feat_list)
            return self.numeric_feat_list + self.bool_feat_list + new_cat_feat_list
        else:
            raise(ValueError("Cleaner must be fitted before new column names can be retrieved"))


    def get_clean_dataframe(self, input_df):
        return pd.DataFrame(self.CleaningPipeline.fit_transform(input_df), columns=self.get_output_col_neams())



def stratified_split_off_validation(input_df, output_dir, name='',validation_size=0.25):
    X = np.array(input_df.iloc[:, :-1])
    y = np.array(input_df['ispos_truth'])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_size)
    for train_idx, validation_idx in sss.split(X, y):
        pass
    # outputs it into the same format as it was input
    input_df.iloc[train_idx, :].to_csv(output_dir + 'train_validation.' + name + '.csv', header=True, index_label='pat_id')
    input_df.iloc[validation_idx, :].to_csv(output_dir + 'test.' + name + '.csv', header=True, index_label='pat_id')

def load_data(data_path):
    """Loaded the data as written by stratified_split_off_validation. This is a little cleaner than the form of load_raw_data so a different reader was required"""

    return pd.read_csv(data_path,index_col='pat_id')


