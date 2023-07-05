from pathlib import Path
from itertools import permutations, combinations, product  # or *_with replacements

import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

from libs import mappings

DATA, LABELS = 0, 1

def get_data(path):

    dirs = [d for d in path.iterdir() if d.is_dir() and 'pc' not in d.stem]
    dirs = sorted(dirs, key=lambda x:int(x.stem[2:]))

    data =   [np.load(d/f'processed_data.npy') for d in sorted(dirs)]
    labels = [np.load(d/f'labels.npy')         for d in sorted(dirs)]
    return data, labels

def get_covs(x):
    x = x.transpose(0, 2, 1)
    return Covariances(estimator='lwf').transform(x)

def transfer_decode(source, target):

    x_train, y_train = source
    x_test, y_test = target

    clf = MDM(metric='kullback_sym')

    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    
    auc = roc_auc_score(y_test, y_pred[:, 1])

    return auc

def transfer_learning(data, labels, components):
    ppts = zip(data, labels)

    ppt_ids = list(mappings.kh_to_ppt().keys())
    ppt_comparisons = list(permutations(ppt_ids, 2))

    aucs = np.empty(len(ppt_comparisons))

    for idx_c, (source, target) in enumerate(permutations(ppts, 2)):
        source, target = list(source), list(target)
        source[DATA], target[DATA] = source[DATA][:, :components, :components], target[DATA][:, :components, :components]

        source[1], target[1] = np.where(source[1]=='move', 1, 0), np.where(target[1]=='move', 1, 0)

        aucs[idx_c] = transfer_decode(source, target)
        
        print(f'{components} | {ppt_comparisons[idx_c]}: {aucs[idx_c].mean():.2f}')

    return aucs

def main(path):

    tasks =   ['grasp', 'imagine']
    filters = ['beta', 'hg', 'betahg']

    # load data outside of loop
    data = {f'{t}_{f}': get_data(path/t/f) for t, f in product(tasks, filters)}

    # Calculate samples covariance matrices
    for k, v in data.items():
        covs = [get_covs(d) for d in v[0]]
        data[k] = list(data[k])
        data[k][0] = covs 

    # components = np.arange(5, 51, 5)[::-1]
    components = [10]

    for npcs in components:

        for task, filter_ in product(tasks, filters):
            
            covs, labels = data[f'{task}_{filter_}']

            labels = [np.where(l=='rest', 'rest', 'move') for l in labels]  # binary

            savepath = Path(f'./transfer_results/pc{npcs}/{task}/{filter_}')
            savepath.mkdir(exist_ok=True, parents=True)

            aucs = transfer_learning(covs, labels, npcs)

            with open(savepath/'transfer_auc.npy', 'wb') as f:
                np.save(f, aucs)


if __name__=='__main__':
    main(Path('results/full_run'))