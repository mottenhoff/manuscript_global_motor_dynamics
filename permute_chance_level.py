from itertools import product
from pathlib import Path
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

def load(path):

    ev, labels, pcs = [], [], []
    
    for folder in path.iterdir():
        
        if not folder.is_dir() or 'kh' not in folder.name:
            continue
        
        ev +=     [np.load(folder/'full_explained_variance.npy')]
        labels += [np.load(folder/'labels.npy')]
        pcs +=    [np.load(folder/'processed_data.npy')]
    
    return ev, labels, pcs

def calculate_chance_level(y, yh, alpha=0.05, n_repetitions=10000):
    np.random.seed(596)
    # y = np.where(y=='rest', 0, 1)
    y = np.where(y=='rest', 1, 0)  # --> LabelEncoder in decoder was wrong when rerunning for y_hat. Fixed in code, but this is a quickfix to no have to rerun the code.

    permuted_scores = np.array([])
    for _ in np.arange(n_repetitions):

        permuted = np.random.permutation(yh)

        auc = roc_auc_score(y, permuted)
        # auc = roc_auc_score(permuted, yh)

        permuted_scores = np.append(permuted_scores, auc)

    chance_idx = int(n_repetitions * (1-alpha))
    chance_level = np.sort(permuted_scores)[chance_idx]

    true_auc = roc_auc_score(y, yh)

    # print(f'auc: {true_auc:.2f}, chance: {chance_level:.2f}')
    # print(chance_level)



    return (chance_level, true_auc), permuted_scores

def plot_chance_levels(n_repetitions):
    scores = np.load('scores.npy')
    permutations = np.load('permutations.npy')

    print(permutations.mean())

    # mean, std = permutations.mean(), permutations.std()
    
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # x = np.linspace(mean - 3*std, mean+3*std, 100)
    # ax[0].plot(x, stats.norm.pdf(x, mean, std))
    # # ax[0].axvline(x[95], linestyle='--')
    
    # for p in permutations:
    #     p = np.sort(p)
    #     ax[1].hist(p, density=True, linewidth=0.1)
    #     ax[1].axvline(p[95], linewidth=0.1)

    # fig.savefig('permutations.png')

    sorted_permutations = np.sort(permutations, axis=1)
    p95 = sorted_permutations[:, int(n_repetitions*.95)]
    print(p95.min(), p95.max(), p95.mean(), p95.std())


    return

def main(path, n_repetitions=100):
    
    predictions = list(Path('results/full_run/').rglob('y_hat.npy'))
    labels = [Path(*file.parts[:-3])/file.parent.name/'labels.npy' for file in predictions]


    scores, permutations = [], np.empty((0, n_repetitions))
    for y, y_hat in zip(labels, predictions):
        # print(y_hat.parent, end=' | ')
        y, y_hat = np.load(y), np.load(y_hat)
        score, permutation = calculate_chance_level(y, y_hat, n_repetitions=n_repetitions)
        scores += [score]
        permutations = np.vstack([permutations, permutation])
        # break


    np.save('scores.npy', scores)
    np.save('permutations.npy', permutations)
    
    return


if __name__=='__main__':
    n_repetitions = 10000

    main_path = Path(r'./results/')
    # main(main_path/'full_run', n_repetitions=10000)
    plot_chance_levels(n_repetitions=10000)