import logging
from dataclasses import dataclass

import numpy as np
import pyriemann
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.pipeline import make_pipeline

from libs.load.load_yaml import load_yaml

TRAIN = 0
TEST = 1

REST, MOVE = 0, 1

TRIALS = 0

logger = logging.getLogger(__name__)
c = load_yaml('./config.yml') 

def split(x, n_splits):
    return np.array(np.vsplit(x, n_splits))

def unsplit(x):
    # Expects [trials x samples x channels]
    return np.vstack(x) 

def save(savepath, **kwargs):

    for name, result in kwargs.items():

        if name == 'train':
            continue

        elif name == 'test':

            with open(savepath/'fold_idc.npy', 'wb') as f:
                np.save(f, np.array([r.fold[TEST] for r in result]))

            with open(savepath/'confusion_matrices.npy', 'wb') as f:
                np.save(f, np.dstack([r.metrics.cm for r in result]).transpose(2, 0, 1))

            if result[0].pca.n_components != 50:
                continue

            evs = np.stack([np.vstack([fold.pca.explained_variance_, 
                                       fold.pca.explained_variance_ratio_]).T
                            for fold in result])
            with open(savepath/'explained_variances.npy', 'wb') as f:
                np.save(f, evs)
            
            principle_axes = np.dstack([fold.pca.components_ for fold in result])
            with open(savepath/'principal_axes.npy', 'wb') as f:
                np.save(f, principle_axes)

        with open(savepath/f'aucs_{name}.npy', 'wb') as f:
            np.save(f, np.array([r.metrics.auc for r in result]))

        predictions = np.vstack([np.hstack([fold.y_hat, fold.y[:, np.newaxis]]) for fold in result])
        with open(savepath/f'predictions_{name}.npy', 'wb') as f:
            np.save(f, predictions)

def get_folds_stratified(y, n_folds):
    # y.shape = [samples x classes] or usually [n x 1]

    shuffle = True if c.decode.shuffle else False

    return StratifiedKFold(n_folds, shuffle=shuffle).split(np.arange(y.shape[0]), y)

def score(y, y_hat):

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

    example_cov = decoder[:1].transform(x_pcs[:1, :, :])
    with open('example_trial.npy', 'wb') as f:
        np.save(f, example_cov)
        
    rest, move = decoder['mdm'].covmeans_
    with open('rest_mean_cov.npy', 'wb') as f:
        np.save(f, rest)

    with open('move_mean_cov.npy', 'wb') as f:
        np.save(f, move)

    return decoder.predict_proba(x_pcs)


def drop_channels(x, dropout):
    
    n_dims = x.shape[-1]  # = [trials x samples x channels]
    x_dropout = x.copy()

    n_chs_to_drop = np.round(n_dims * (1-dropout)).astype(int)

    if n_chs_to_drop < 1:
        # print(f'Not enough channels to drop. Skipping {dropout}')
        return x_dropout, np.empty(0)

    chs_to_drop = np.random.choice(np.arange(n_dims), n_chs_to_drop, replace=False)

    x_dropout[:, :, chs_to_drop] = 0

    return x_dropout, chs_to_drop

def decode_dropout(session, savepath):

    components = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    dropouts =   [1, .9, .8, .7, .6, .5]

    trial_labels = np.where(session.trial_labels=='rest', REST, MOVE)

    n_classes =  2
    n_dropouts = len(dropouts)
    n_repeats =  10
    n_folds =    10
    n_metrics =  2

    x = session.trials
    y = np.where(session.trial_labels=='rest', REST, MOVE)

    full_results =               np.empty((n_folds, n_metrics,  2))  # 2 = [train, test]
    full_confusion_matrices =    np.empty((n_folds, n_classes,  n_classes))  # Only test
    dropout_results =            np.full(( n_folds, n_dropouts, n_repeats, n_metrics), np.nan)  # Only for test
    dropout_confusion_matrices = np.full(( n_folds, n_dropouts, n_classes, n_classes), np.nan)  # Only test

    for component in components:

        path = savepath/f'pc{component}'/f'{session.ppt_id}'
        path.mkdir(parents=True, exist_ok=True)

        y_hat = np.empty(y.shape)
        for i_fold, fold in enumerate(get_folds_stratified(trial_labels, n_folds)):

            x_test, x_train = x[fold[TEST], :, :], x[fold[TRAIN], :, :]
            y_test, y_train = y[fold[TEST]],       y[fold[TRAIN]]

            # Fit and transform
            pca, decoder = fit_decoder(x_train, y_train, component)
            y_hat_train =  run_decoder(x_train, pca, decoder)
            y_hat_test =   run_decoder(x_test, pca, decoder)

            y_hat[fold[TEST]] = y_hat_test[:, 1]

            scores_train, scores_test = score(y_train, y_hat_train), score(y_test,  y_hat_test)

            logger.info(f'{session.ppt_id:<5s} | pc: {component:<3d} | Train: {scores_train[0]:<5.2f} | Test: {scores_test[0]:<5.2f}')

            full_results[i_fold, :, :] = np.vstack([scores_train[:2], scores_test[:2]]).T
            full_confusion_matrices[i_fold, :, :] = scores_test[-1]

            if c.decode.dropout:
                for i_dropout, dropout in enumerate(dropouts):

                    for repeat in np.arange(n_repeats):

                        x_test_dropout, dropped_chs = drop_channels(x_test, dropout)

                        info_str = f'{component:>3} - {i_fold:>3} - {dropout:>3} - {repeat:>3}' \
                                    + (' - no channels dropped)') if dropped_chs.size==0 else f'{dropped_chs}'
                        logger.info(info_str)

                        y_hat_dropout = run_decoder(x_test_dropout, pca, decoder)
                        scores_dropout = score(y_test, y_hat_dropout)
                        
                        dropout_results[i_fold, i_dropout, repeat, :] =       scores_dropout[:2]
                        dropout_confusion_matrices[i_fold, i_dropout, :, :] = scores_dropout[-1]

        with open(path/'y_hat.npy', 'wb') as f:
            np.save(f, y_hat)

        with open(path/'full.npy', 'wb') as f:
            np.save(f, full_results)

        if c.decode.dropout:
            with open(path/'dropout.npy', 'wb') as f:
                np.save(f, dropout_results)

    return