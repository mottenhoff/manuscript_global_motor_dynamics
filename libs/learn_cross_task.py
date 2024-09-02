import logging
from itertools import permutations

import numpy as np
import pyriemann
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline

from libs.load.load_yaml import load_yaml

GRASP, IMAGINE = 0, 1
TRAIN, TEST = 0, 1

REST = 0
MOVE = 1

TRIALS = 0

logger = logging.getLogger(__name__)
c = load_yaml('./config.yml') 

def split(x, n_splits):
    return np.array(np.vsplit(x, n_splits))

def unsplit(x):
    # Expects [trials x samples x channels]
    return np.vstack(x) 

def evaluate(y, y_hat):
    return roc_auc_score(y, y_hat[:, MOVE]), \
           balanced_accuracy_score(y, np.argmax(y_hat, axis=1)), \
           confusion_matrix(y, np.argmax(y_hat, axis=1))

def pca_fit(n_components, x):
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=n_components))
    return pca.fit(unsplit(x))

def fit_decoder(x, y, n_components):

    # Pipelines
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=n_components))

    # Expects [Trials x Channels x time]
    decoder = make_pipeline(pyriemann.estimation.Covariances(estimator='lwf'),
                            pyriemann.classification.MDM(metric='kullback_sym'))

    # Fitting
    x_shape = x.shape

    x_stacked = unsplit(x.copy())
    
    x_pcs = pca.fit_transform(x_stacked)
    
    # Unstack?
    x_pcs = split(x_pcs, x_shape[TRIALS])
    
    x_pcs = x_pcs.transpose(0, 2, 1)

    decoder.fit(x_pcs, y)

    return pca, decoder

def run_decoder(x, pca, decoder):

    x_shape = x.shape

    x_stacked = unsplit(x.copy())
    
    x_pcs_stacked = pca.transform(x_stacked)

    x_pcs = split(x_pcs_stacked, x_shape[TRIALS]).transpose(0, 2, 1)

    return decoder.predict_proba(x_pcs)

def drop_channels(x, dropout):
    
    n_dims = x.shape[-1]  # = [trials x samples x pcs]
    x_dropout = x.copy()

    n_chs_to_drop = np.round(n_dims * (1-dropout)).astype(np.int)
    
    if n_chs_to_drop < 1:
        return x_dropout, np.empty(0)

    chs_to_drop = np.random.choice(np.arange(n_dims), n_chs_to_drop, replace=False)

    x_dropout[:, :, chs_to_drop] = 0

    return x_dropout, chs_to_drop

def decode(train, test, n_components):

    x_test, x_train = test.trials, train.trials
    y_train = train.trial_labels

    pca, decoder = fit_decoder(x_train, y_train, n_components)

    y_hat_train = run_decoder(x_train, pca, decoder)
    y_hat_test =  run_decoder(x_test, pca, decoder)

    return y_hat_train, y_hat_test

def decode_cross(sessions, savepath):

    components = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    n_components = len(components)
    n_classes = 2
    n_metrics = 2

    logger.info(f'Saving results to {savepath}')

    for session in sessions:
        session.trial_labels = np.where(session.trial_labels=='rest', REST, MOVE)

    for train, test in permutations(sessions):
        
        logger.info(f'\n{train.name} to {test.name}')

        results_train = np.empty((n_components, n_metrics))
        results_test =  np.empty((n_components, n_metrics))
        cms_train =     np.empty((n_components, n_classes, n_classes))
        cms_test =      np.empty((n_components, n_classes, n_classes))
        
        for idx_components, num_components in enumerate(components):
            
            y_hat_train, y_hat_test = decode(train, test, num_components)
            auc_train, bac_train, cm_train = evaluate(train.trial_labels, y_hat_train)
            auc_test, bac_test, cm_test = evaluate(test.trial_labels,  y_hat_test)

            results_train[idx_components, :] = [auc_train, bac_train]
            results_test[idx_components, :] =  [auc_test, bac_test]
            cms_train[idx_components, :, :] = cm_train
            cms_test[idx_components, :, :] = cm_test

            logger.info(f'{train.ppt_id:<2s} | pc: {num_components:<3d} | Train: {auc_train:<5.2f} ({train.name}) | Test: {auc_test:<5.2f} ({test.name})')

        path = savepath/train.ppt_id/f'{train.name}-to-{test.name}'
        path.mkdir(parents=True, exist_ok=True)

        np.save(path/'results_train.npy', results_train)
        np.save(path/'results_test.npy', results_test)
        np.save(path/'confusion_matrices_train.npy', cms_train)
        np.save(path/'confusion_matrices_test.npy', cms_test)