from copy import copy

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from libs.load.load_yaml import load_yaml
from libs.split_to_trials import c2t

c = load_yaml('./config.yml')

def explore(session, savepath):
    
    session = copy(session)

    if c.explore.pca:
        pca = make_pipeline(StandardScaler(),
                            PCA(n_components=50))

        session.eeg = pca.fit_transform(session.eeg)

        session = c2t(session)
        
        with open(savepath/session.ppt_id/f'processed_data.npy', 'wb') as f:
            np.save(f, session.trials)

        # with open(savepath/session.ppt_id/f'full_eigen_vectors.npy', 'wb') as f:
            # np.save(f, pca.named_steps['pca'].components_)

        with open(savepath/session.ppt_id/f'full_explained_variance.npy', 'wb') as f:
            np.save(f, np.vstack([pca.named_steps['pca'].explained_variance_, 
                                  pca.named_steps['pca'].explained_variance_ratio_]).T)
   
        # with open(savepath/session.ppt_id/f'processed_data.npy', 'wb') as f:
        #     np.save(f, session.trials)
        
        with open(savepath/session.ppt_id/f'labels.npy', 'wb') as f:
            np.save(f, session.trial_labels)