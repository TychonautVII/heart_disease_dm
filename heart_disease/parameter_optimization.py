import numpy as np
import pandas as pd
import logging

from sklearn.pipeline import Pipeline

from sklearn.linear_model.perceptron import Perceptron
from sklearn.mixture import GaussianMixture as GM

# from sklearn.decomposition import PCA
from heart_disease.preprocessing import pass_through_PCA as PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

import heart_disease.preprocessing as hdpp
from heart_disease.globals import data_path, output_path


logger = logging.getLogger(__name__)


algorithum_list = ["SVC", "SVC_poly", "GM", "Perceptron_PCA", "Perceptron_LDA"]


def __grid_search_wrapper(model, parameters, X, y, name='',
                          n_jobs=4, test_size=0.25, cv=None, n_splits=10):
    """A wrapper around sklearns GridSearchCV for my convience"""
    if cv is None:
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

    cv_estimator = GridSearchCV(model, parameters, cv=cv, n_jobs=n_jobs, scoring='accuracy')
    logger.info("{} Grid Search Started".format(name))
    out = cv_estimator.fit(X, y)
    logger.info("{} Grid Search Complete".format(name))
    logger.info("{} Best Score: {}".format(name, out.best_score_))
    logger.info("{} Best Params: {}".format(name, out.best_params_))

    return pd.DataFrame(out.cv_results_)

def get_paramater_grids(data_path):
    """Returns all the Models / Paramters I'm searching over"""

    pca_nfeatures = [0, 1, 2, 5, 10, 15, 20, 24]
    # 0 is a pass through value that does not do PCA
    # 24 is safer because some features might be found in the training set. This might not be the safest design, but it works for now


    # some may have more or few features based on training. 24 seems safe

    # Number of training points allowed to be on the wrong side of hyperplane.
    # A point is fractionally over the line if it violates the margin
    # 0 is a lower bound
    # The approximent number of training points (170)/2 seems like a reasonable upper bound (n
    # The sklearn formulation is different than what was covered in the book
    svc_gamma = np.logspace(-7, 0, 8) #kernal parameter

    paramater_grids = {}

    # SVC
    paramater_grids['SVC'] = {}
    paramater_grids['SVC']['pipeline']=\
        Pipeline([
            ('cleaner', hdpp.DataCleaner(data_path + 'meta_data.csv').CleaningPipeline),
            ('feature', PCA()),
            ('classifier', SVC())
            ])
    paramater_grids['SVC']['parameters'] = \
        {'classifier__kernel':('linear','rbf','sigmoid'), #Linear doesn't gues gamma, but whatevs
         'classifier__C':  np.logspace(-2,4,7),
         'classifier__gamma': svc_gamma,
         'feature__n_components': pca_nfeatures}

    # SVC with a polynomial kernal
    paramater_grids['SVC_poly'] = {}
    paramater_grids['SVC_poly']['pipeline']=\
        Pipeline([
            ('cleaner', hdpp.DataCleaner(data_path + 'meta_data.csv').CleaningPipeline),
            ('feature', PCA()),
            ('classifier', SVC(kernel='poly'))
            ])
    paramater_grids['SVC_poly']['parameters'] = \
        {'classifier__degree': [2, 3, 4, 5],
         'classifier__C':  np.logspace(-2,3,6),
         'classifier__gamma': svc_gamma,
         'feature__n_components': pca_nfeatures}

    # Gaussian Mixture
    paramater_grids['GM'] = {}
    paramater_grids['GM']['pipeline']=\
        Pipeline([
            ('cleaner', hdpp.DataCleaner(data_path + 'meta_data.csv').CleaningPipeline),
            ('feature', PCA()),
            ('classifier', GM(max_iter=250,n_init=25))
            ])
    paramater_grids['GM']['parameters'] = \
        {'classifier__n_components': [1, 2, 4, 8, 16, 32],
         'classifier__covariance_type': ['full', 'tied', 'diag', 'spherical'],
         'feature__n_components': pca_nfeatures}

    # Perceptron
    paramater_grids['Perceptron_PCA'] = {}
    paramater_grids['Perceptron_PCA']['pipeline'] = \
        Pipeline([
            ('cleaner', hdpp.DataCleaner(data_path + 'meta_data.csv').CleaningPipeline),
            ('feature', PCA()),
            ('classifier', Perceptron())
        ])
    paramater_grids['Perceptron_PCA']['parameters'] = \
        {'feature__n_components': pca_nfeatures}

    # Perceptron
    paramater_grids['Perceptron_LDA'] = {}
    paramater_grids['Perceptron_LDA']['pipeline'] = \
        Pipeline([
            ('cleaner', hdpp.DataCleaner(data_path + 'meta_data.csv').CleaningPipeline),
            ('feature', LinearDiscriminantAnalysis()),
            ('classifier', Perceptron())
        ])
    paramater_grids['Perceptron_LDA']['parameters'] = {}

    return paramater_grids


def execute_grid_search(X,y,algs_2_run=None,n_splits=25,n_jobs=4):

    if algs_2_run is None:
        algs_2_run = algorithum_list

    search_dict = get_paramater_grids(data_path)

    for name, details_dict in search_dict.items():
        if name in algs_2_run:
            results = __grid_search_wrapper(details_dict['pipeline'],details_dict['parameters'],X,y, name=name,n_splits=n_splits, n_jobs=n_jobs)
            results.to_pickle(output_path + name + '.grid_search.pkl')
        else:
            logging.info("{} Grid Search Skipped".format(name))

def load_grid_search_summary():
    cols2keep = ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "params"]
    reduced_frame_list = []
    for algorithum in algorithum_list:
        results_df = pd.read_pickle(output_path + algorithum + '.grid_search.pkl')
        results_df = results_df.loc[:, cols2keep]
        results_df['algorithum'] = algorithum
        reduced_frame_list.append(results_df)
        if algorithum == 'SVC_poly':
            results_df['params'].apply(lambda x: x.update({'classifier__kernel': 'poly'}))
    # Final Processing
    summary_df = pd.concat(reduced_frame_list)
    summary_df['sigma_low_test_score'] = summary_df['mean_test_score'] - summary_df['std_test_score']
    summary_df['sigma_low_train_score'] = summary_df['mean_train_score'] - summary_df['std_train_score']
    summary_df = summary_df.sort_values(by='sigma_low_test_score', ascending=False)
    # TODO Maybe this sorting could be made more complicated meaningfully. We could truncate the primary metric at
    # a meaningful amout of sigfigs, and sort by a secondard metric to establish the rankings
    summary_df['rank'] = np.array(range(summary_df.shape[0]))

    return summary_df

