import numpy as np
import pandas as pd
import logging

from sklearn.pipeline import Pipeline

from sklearn.linear_model.perceptron import Perceptron
from sklearn.mixture import GaussianMixture as GM
from sklearn.svm import SVC

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

import heart_disease.preprocessing as hdpp
from heart_disease.globals import data_path, output_path


logger = logging.getLogger(__name__)

def __grid_search_wrapper(model, parameters, X, y, name='',
                          n_jobs=1, test_size=0.25, cv=None, n_splits=10):
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

    pca_nfeatures = [1, 2, 5, 10, 15, 20, 25]
    svc_C = [0.5, 1, 3.5, 10, 15]
    svc_gamma = np.logspace(-6, -0.1, 10)

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
        {'classifier__kernel':('linear','poly','rbf','sigmoid'),
         'classifier__C': svc_C,
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
         'classifier__C': svc_C,
         'classifier__gamma': svc_gamma,
         'feature__n_components': pca_nfeatures}

    # Gaussian Mixture
    paramater_grids['GM'] = {}
    paramater_grids['GM']['pipeline']=\
        Pipeline([
            ('cleaner', hdpp.DataCleaner(data_path + 'meta_data.csv').CleaningPipeline),
            ('feature', PCA()),
            ('classifier', GM())
            ])
    paramater_grids['GM']['parameters'] = \
        {'classifier__n_components': [1, 2, 4, 8, 16, 32],
         'classifier__covariance_type': ['full', 'tied', 'diag', 'spherical'],
         'feature__n_components': pca_nfeatures}

    # Perceptron
    paramater_grids['Perceptron'] = {}
    paramater_grids['Perceptron']['pipeline'] = \
        Pipeline([
            ('cleaner', hdpp.DataCleaner(data_path + 'meta_data.csv').CleaningPipeline),
            ('feature', PCA()),
            ('classifier', Perceptron())
        ])
    paramater_grids['Perceptron']['parameters'] = \
        {'feature__n_components': pca_nfeatures}

    return paramater_grids


def execute_grid_search(X,y):
    search_dict = get_paramater_grids(data_path)

    for name, details_dict in search_dict.items():
        results = __grid_search_wrapper(details_dict['pipeline'],details_dict['parameters'],X,y, name=name,n_splits=1)
        results.to_pickle(output_path + name + '.grid_search.pkl')



