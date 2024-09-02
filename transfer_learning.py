from pathlib import Path
from itertools import permutations, product

import numpy as np
from sklearn.metrics import roc_auc_score
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

from libs import mappings

DATA, LABELS = 0, 1
REST, MOVE = 0, 1

def estimate_covariance_matrices(x):
    x = x.transpose(0, 2, 1)
    return Covariances(estimator='lwf').transform(x)

def load_covariance_matrices(path):
    sort_by_parent_dir = lambda p: p.parent.name

    data =   [np.load(file) for file in sorted(path.rglob('processed_data.npy'), key=sort_by_parent_dir)]
    labels = [np.load(file) for file in sorted(path.rglob('labels.npy'),         key=sort_by_parent_dir)]

    covs = [estimate_covariance_matrices(cov) for cov in data]
    labels = [np.where(l != 'rest', MOVE, REST) for l in labels]

    return covs, labels

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

    # ppt_ids = list(mappings.kh_to_ppt().keys())
    ppt_ids = mappings.PPTS
    ppt_comparisons = list(permutations(ppt_ids, 2))

    aucs = np.empty(len(ppt_comparisons))

    for idx_c, (source, target) in enumerate(permutations(ppts, 2)):
        source, target = list(source), list(target)
        source[DATA], target[DATA] = source[DATA][:, :components, :components], target[DATA][:, :components, :components]

        aucs[idx_c] = transfer_decode(source, target)
        
        print(f'{components} | {ppt_comparisons[idx_c]}: {aucs[idx_c].mean():.2f}')

    return aucs

def main(path):

    tasks =   ['grasp', 'imagine']
    filters = ['beta', 'hg', 'betahg']
    components = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for npcs in components:

        for task, filter_ in product(tasks, filters):
            
            print(task, filter_)
            covs, labels = load_covariance_matrices(path/task/filter_)

            savepath = Path(f'./transfer_results/pc{npcs}/{task}/{filter_}')
            savepath.mkdir(exist_ok=True, parents=True)

            aucs = transfer_learning(covs, labels, npcs)

            with open(savepath/'transfer_auc.npy', 'wb') as f:
                np.save(f, aucs)


if __name__=='__main__':
    main(Path('results/full_run'))