from itertools import product, combinations, combinations_with_replacement
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.spatial import procrustes

from scipy.linalg import qr, svd, inv

ABSOLUTE, RELATIVE, CUMULATIVE = 0, 1, 2
PCS = 1
TRIAL_LENGTH = 3
COLOR_REST = cm.plasma(0)
COLOR_MOVE = cm.plasma(127)
# COLOR_MOVE = cm.plasma(255)
EV_ABSOLUTE, EV_RELATIVE, EV_CUMULATIVE = 0, 1, 2


def canoncorr(X:np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    From https://github.com/BeNeuroLab/2022-preserved-dynamics/blob/main/tools/ccaTools.py#L290

    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    
    returns: A,B,r,U,V 
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X,0)
    Y = Y - np.mean(Y,0)

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1,T11,perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0,0])))).eps*max([n,p1]))

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:,:rankX]
        T11 = T11[rankX,:rankX]

    Q2,T22,perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0,0])))).eps*max([n,p2]))

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:,:rankY]
        T22 = T22[:rankY,:rankY]

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX,rankY)
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:,:d] * np.sqrt(n-1)
    B = inv(T22) @ M[:,:d] * np.sqrt(n-1)
    r = D[:d]
    # remove roundoff errs
    r[r>=1] = 1
    r[r<=0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1,:] = np.vstack((A, np.zeros((p1-rankX,d))))
    B[perm2,:] = np.vstack((B, np.zeros((p2-rankY,d))))
    
    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V

def load(path):

    ev, labels, pcs = [], [], []

    ids = []
    
    for folder in path.iterdir():
        
        if not folder.is_dir() or 'kh' not in folder.name:
            continue
        
        ids += [folder.name]
        ev +=     [np.load(folder/'full_explained_variance.npy')]
        labels += [np.load(folder/'labels.npy')]
        pcs +=    [np.load(folder/'processed_data.npy')]
    
    return ev, labels, pcs

def plot_ev(ev, ax=None, name=None):
    # pca.components_ = eigenvectors
    # pca.explained_variance_ = eigenvalues

    n_cols = 3
    titles = ['abs', 'rel', 'cumsum']
    ev = np.array(ev)

    x_ticks = np.arange(ev.shape[PCS])

    # ev_cs = np.cumsum(ev[:, :, 1], axis=1)
    ev = np.dstack([ev, np.cumsum(ev[:, :, 1], axis=1)])
    scree_idx = np.where(ev[:, :, ABSOLUTE].mean(axis=0)<1)[0][0]

    try:
        scree_idx_individual = [np.where(ev[i_ppt, :, ABSOLUTE] <= 1)[0][0] for i_ppt in np.arange(ev.shape[0])]
        print(f'Scree idx: {scree_idx}\n{scree_idx_individual}\n{np.mean(scree_idx_individual)}')
    except IndexError:
        print(scree_idx)


    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))

    for i_col in range(n_cols):

        ax[i_col].step(x_ticks,  ev[:, :, i_col].T, alpha=0.3)

        # ax[i_col].set_title(titles[i_col])
        
        if i_col == 0:
            ax[i_col].axhline(1, linestyle='--', color='black', label='Scree line')
            ax[i_col].set_ylabel('Explained variance')

        if i_col == 1:
            ax[i_col].set_xlabel('Principal components')
            ax[i_col].set_ylabel('Explained variance [%]')
            ax[i_col].set_yticks(np.arange(0, 101, 20)/100)
            ax[i_col].set_yticklabels(np.arange(0, 101, 20)/100)

        if i_col == 2:
            ax[i_col].set_ylabel('Explained variance [Cumulative %]')

        if i_col < 2:            
            ax[i_col].set_yscale('log')

        mean, std = ev[:, :, i_col].mean(axis=0), ev[:, :, i_col].std(axis=0)
        ax[i_col].plot(mean, color='black')
        ax[i_col].fill_between(x_ticks, mean-std, mean+std, alpha=0.2, color='black')
        
        ax[i_col].axvline(scree_idx, color='black', linestyle='--')

        ax[i_col].set_xticks(np.arange(0, 51, 10))

    if name:
        fig.suptitle(name)


    fig.tight_layout()
    fig.savefig(f'./figures/figure_5_ev_{name}.png')

    return fig, ax

def plot_evs(ev, ax=None):

    create_new_axis = True if ax is None else False

    if create_new_axis:

        fig, ax = plt.subplots(nrows=1, ncols=1)

    ev = np.array(ev)

    x_ticks = np.arange(ev.shape[PCS])

    ev = np.dstack([ev, np.cumsum(ev[:, :, 1], axis=1)])
    scree_idx = np.where(ev[:, :, ABSOLUTE].mean(axis=0)<1)[0][0]

    type_ = EV_RELATIVE
    type_ = EV_CUMULATIVE
    mean, std = ev[:, :, type_].mean(axis=0), ev[:, :, type_].std(axis=0)

    ax.step(x_ticks, ev[:, :, type_].T, alpha=.4)
    ax.plot(x_ticks, mean, color='black')
    ax.fill_between(x_ticks, mean-std, mean+std, alpha=0.2, color='black')

    ax.axvline(scree_idx, color='black', linestyle='--', linewidth=1)
    
    x_ticks = np.sort(np.append(np.arange(0, 51, 10), scree_idx))  # insert Scree tick
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize='x-large')
    
    if scree_idx in [22]:
        # Hardcoded adjustment for one overlapping xtick label
        ax.get_xticklabels()[3].set_y(ax.get_xticklabels()[5].get_position()[1] - 0.06)

    # ax.set_yscale('log')
    # ax.set_ylim(0.001, 1)

    # ax.set_yticks([1, 0.1, 0.01, 0.001])
    # formatter = ScalarFormatter()
    # formatter.set_scientific(False)
    # ax.yaxis.set_major_formatter(formatter)
    # ax.set_yticklabels([100, 10, 1, 0.1], fontsize='large')

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])
    ax.spines[['top', 'right']].set_visible(False)


    if create_new_axis:
        return fig, ax
    
    return ax

def plot_pcs(pcs, labels, name=None):

    # Perhaps plot all in one via procrustus?
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html

    N_COMPONENTS = 10

    for idx, (pc, label) in enumerate(zip(pcs, labels)):
    
        fig, axs = plt.subplots(nrows=N_COMPONENTS, ncols=N_COMPONENTS, figsize=(10, 10))
        
        for ax in axs.flatten(): 
            ax.axis('off')

        rest, move = pc[label=='rest'].mean(axis=0), pc[label!='rest'].mean(axis=0)

        rest -= rest[0, :]
        move -= move[0, :]

        for y_idx, x_idx in combinations_with_replacement(np.arange(N_COMPONENTS), 2):

            axs[x_idx, y_idx].axis('on')
            axs[x_idx, y_idx].spines[['top', 'right']].set_visible(False)
            axs[x_idx, y_idx].set_xticks([])
            axs[x_idx, y_idx].set_yticks([])

            if y_idx == 0:
                axs[x_idx, y_idx].set_ylabel(f'PC {x_idx+1}', fontsize='x-large')

            if x_idx == N_COMPONENTS-1:
                axs[x_idx, y_idx].set_xlabel(f'PC {y_idx+1}', fontsize='x-large')

            if x_idx == y_idx:

                axs[x_idx, y_idx].plot(rest[:, x_idx], color=COLOR_REST, linewidth=1)
                axs[x_idx, y_idx].plot(move[:, x_idx], color=COLOR_MOVE, linewidth=1)

                continue
            
            axs[x_idx, y_idx].plot(rest[:, x_idx], rest[:, y_idx], color=COLOR_REST, linewidth=1)
            axs[x_idx, y_idx].plot(move[:, x_idx], move[:, y_idx], color=COLOR_MOVE, linewidth=1)

        fig.suptitle(f'{name} | Participant {idx+1}', fontsize=30)

        fig.tight_layout()

        fig.savefig(f'./figures/pcs/figure_5_pcs_{name}_ppt{idx}.png')


    return fig, ax

def plot_pc_space_avg(path):

    tasks = ('grasp', 'imagine')
    filters = ('beta', 'hg', 'betahg')

    data = [load(path/task/filter) for task, filter in product(tasks, filters)]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axs = axs.flatten()
    
    for idx_cond, condition in enumerate(data):
        
        pcs = []
        for idx_pc, pc in enumerate(condition[2]):
            pc = pc[:, 0:6*1024:2, :] if pc.shape[1] > 3100 else pc[:, 0:3*1024, :]
            rest_idc = condition[1][idx_pc] == 'rest'

            move, rest = pc[~rest_idc, :, :].mean(axis=0, keepdims=True), pc[rest_idc, :, :].mean(axis=0, keepdims=True)

            pc = np.vstack([rest, move])
            pcs += [pc]
        
        pcs = np.stack(pcs)

        pcs -= pcs[:, :, :1 :]

        # for ppt in pcs:
        #     axs[idx_cond].plot(ppt[0, :, 0], ppt[0, :, 1], linewidth=0.1, color=COLOR_REST)
        #     axs[idx_cond].plot(ppt[1, :, 0], ppt[1, :, 1], linewidth=0.1, color=COLOR_MOVE)

        axs[idx_cond].plot(pcs[:, 0, :, 0].mean(axis=0), pcs[:, 0, :, 1].mean(axis=0), linewidth=2, color=COLOR_REST, label='Rest')
        axs[idx_cond].plot(pcs[:, 1, :, 0].mean(axis=0), pcs[:, 1, :, 1].mean(axis=0), linewidth=2, color=COLOR_MOVE, label='Move')

        axs[idx_cond].spines[['top', 'right']].set_visible(False)
        
        xticks = [-1, 0, 1]
        axs[idx_cond].set_xticks(xticks)
        axs[idx_cond].set_yticks(xticks)
        axs[idx_cond].set_xticklabels(xticks, fontsize='x-large')
        axs[idx_cond].set_yticklabels(xticks, fontsize='x-large')

    
    axs[0].set_ylabel('Execute\n\n', fontsize=22)
    axs[3].set_ylabel('Imagine\n\nPC 2', fontsize=22)
    axs[3].set_xlabel('PC 1', fontsize='xx-large')

    axs[0].set_title('Beta', fontsize=22)
    axs[1].set_title('High-gamma', fontsize=22)
    axs[2].set_title('Both', fontsize=22)
    # axs[3].set_xlabel('PC 1', fontsize='xx-large')

    axs[2].legend(bbox_to_anchor=(1.1, 1), frameon=False, fontsize='xx-large')

    fig.tight_layout()
    # fig.savefig('./figures/figure_5_principal_component_space_individual_trajectories.png')
    # fig.savefig('./figures/figure_5_principal_component_space_individual_trajectories.svg')
    fig.savefig('./figures/figure_5_principal_component_space.png')
    fig.savefig('./figures/figure_5_principal_component_space.svg')

    return 

def plot_pc_characterization(pcs, labels, name=None, ax=None):

    create_new_axis = True if ax is None else False

    if create_new_axis:

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))


    for idx, (pc, label) in enumerate(zip(pcs, labels)):
        
        rest, move = pc[label=='rest'].mean(axis=0), pc[label!='rest'].mean(axis=0)
    
        rest -= rest[0, :]
        move -= move[0, :]

        ax.plot(np.linspace(0, TRIAL_LENGTH, rest.shape[0]), rest[:, 0], linewidth=0.5, alpha=0.5, color=COLOR_REST)
        ax.plot(np.linspace(0, TRIAL_LENGTH, move.shape[0]), move[:, 0], linewidth=0.5, alpha=0.5, color=COLOR_MOVE)


        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([0, 1, 2, 3], fontsize='x-large')
        ax.set_xlim(0, 3)
        
        ax.set_yticks([0])
        ax.set_yticklabels([0], fontsize='x-large')


    # ax.set_title(f'{name}')
    # fig.suptitle(f'{name}')

    if create_new_axis:

        fig.tight_layout()
        fig.savefig(f'./figures/figure_5_first_component.png')

        return fig, ax

    return ax

def plot_avg_from_ax(ax, idx, legend=False):

    lines = [line for line in ax.get_lines()]

    rest_idc = [idx_rest for idx_rest, line in enumerate(lines) if line._color == COLOR_REST]
    move_idc = [idx_move for idx_move, line in enumerate(lines) if line._color == COLOR_MOVE]

    lines = [line.get_ydata() for line in lines]

    lines = [line[0:len(line):2] if len(line) > 3100 else line for line in lines]
    lines = [line[:3*1024] for line in lines]
    
    lines = np.array(lines)
    rest, move = lines[rest_idc], lines[move_idc]
    
    ax.plot(np.linspace(0, 3, 3*1024), np.mean(rest, axis=0), linewidth=2, color=COLOR_REST, label='Rest' if legend else None)
    ax.plot(np.linspace(0, 3, 3*1024), np.mean(move, axis=0), linewidth=2, color=COLOR_MOVE, label='Move' if legend else None)

    if legend:
        ax.legend(fontsize='xx-large', bbox_to_anchor=(1,1), frameon=False)

    return ax

def canonical_correlation(pcs, labels, name):
    # Gallego et al. alligned via CCA, but that requires a trial structure
    # Perhaps I can allign with procrustus or something? First try unaligned.

    # map to single matrix
    n_ppts = len(pcs)

    pcs = np.array([pc[:118, :3*1024, :] if pc.shape[1]<3100 else pc[:118, :2*3*1024:2, :] for pc in pcs])
    labels = np.array([label[:118] for label in labels])

    # TODO: Selecting on class seems to contain an error
    is_rest_trial = labels=='rest'
    pcs = np.stack([pcs[:, ~is_rest_trial[idx], :, :] for idx in range(len(is_rest_trial))])
    pcs = np.reshape(pcs, (pcs.shape[0], -1, pcs.shape[-1]))

    n_components = 10
    pcs = pcs[:, :, :n_components]  # Only first 10 for now

    ccs = np.empty((0, 3, n_components))

    for ppt_a, ppt_b in combinations(range(n_ppts),  2):
        # Unaligned
        cc = np.corrcoef(pcs[ppt_a], pcs[ppt_b], rowvar=False)[range(n_components), range(n_components, 2*n_components)]

        # CCA aligned
        cca = canoncorr(pcs[ppt_a], pcs[ppt_b])
    
        # Procrustes aligned
        m1, m2, d = procrustes(pcs[ppt_a], pcs[ppt_b])
        ccp = np.corrcoef(m1, m2, rowvar=False)[range(n_components), range(n_components, 2*n_components)]

        ccs = np.vstack([ccs, np.vstack([cc, cca, ccp])[np.newaxis, :, :]])
        

    mean, std = ccs.mean(axis=0), ccs.std(axis=0)


    titles = ('Correlation', 'Canonical', 'Procrustus')
    fig, ax = plt.subplots(nrows=1, ncols=3)
    for idx in range(ccs.shape[1]):
        m, s = mean[idx, :], mean[idx, :]

        ax[idx].plot(m, color='black')
        ax[idx].fill_between(np.arange(m.shape[-1]), m-s, m+s, alpha=0.2, color='black')

        ax[idx].spines[['top', 'right']].set_visible(False)
        ax[idx].set_xticks(np.arange(n_components))
        ax[idx].set_xticklabels(np.arange(n_components)+1)
        # ax[idx].set_ylim(-.25, .25)

        ax[idx].set_title(f'{titles[idx]}')

        if idx == 0:
            ax[idx].set_ylabel('CC', fontsize='x-large')

        if idx == 1:
            ax[idx].set_xlabel('Principal component', fontsize='x-large')
        
    fig.savefig('./figures/figure_5_canonical_correlation_analysis_move.png')
    
    return






    # # Only for first component now
    # components = 0
    # for ppt_A, ppt_B in combinations(range(n_ppts), 2):
        
    #     cc_unaligned = np.corrcoef(pcs[ppt_A, :, components], pcs[ppt_B, :, components])[1, 0]

    #     # CCA
    #     # cca = CCA(n_components = pcs.shape[1])
    #     # cca.fit
        
    #     # Procrustus


    #     print(f'{ppt_A}-{ppt_B}: {cc_unaligned:.3f}, {cc_procrustes:.3f}, disparity: {d:.3f}')

    # # move, rest = pcs[:, ~is_rest_trial, :], pcs[is_rest_trial, :]

    # print()


    return



def make(path: Path):

    plot_pc_space_avg(path)
    
    tasks = ('grasp', 'imagine')
    filters = ('beta', 'hg', 'betahg')

    fig_pcs, axs_pcs = plt.subplots(nrows=len(tasks), ncols=len(filters), figsize=(16, 9))
    axs_pcs = axs_pcs.flatten()

    fig_evs, axs_evs = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
    axs_evs = axs_evs.flatten()
    
    
    for idx, (task, filter) in enumerate(product(tasks, filters)):
        name = f'{task}_{filter}'

        ev, labels, pcs = load(path/task/filter)

        # canonical_correlation(pcs, labels, name=name)
        # axs_evs[idx] = plot_evs(ev, ax=axs_evs[idx])

        fig, ax = plot_pcs(pcs, labels, name=name)
        # axs_pcs[idx] = plot_pc_characterization(pcs, labels, name=name, ax=axs_pcs[idx])

    # for idx, ax in enumerate(axs_pcs):
    #     ax = plot_avg_from_ax(ax, idx, legend=True if idx == 2 else False)

    # axs_pcs[3].set_xlabel('Trial time [s]', fontsize='xx-large')
    # axs_pcs[3].set_ylabel('Value first component', fontsize='xx-large')

    axs_evs[3].set_xlabel('Principal Components', fontsize='xx-large')
    axs_evs[3].set_ylabel('Explained Variance [%]', fontsize='xx-large')
    axs_evs[0].annotate('Scree line', xy=(22/50, 1.03), xycoords='axes fraction', ha='center', va='center', fontsize='large')

    # axs_evs[0].set_ylabel('Execute\n\n', fontsize='xx-large')
    # axs_evs[3].set_ylabel('Imagine\n', fontsize=22)
    # axs_evs[3].set_xlabel('PC 1', fontsize='xx-large')
    pad = 20
    fs = 24
    axs_evs[0].set_title('Beta', fontsize=fs, pad=pad)
    axs_evs[1].set_title('High-gamma', fontsize=fs, pad=pad)
    axs_evs[2].set_title('Both', fontsize=fs, pad=pad)

    axs_evs[4].set_xlabel('Execute', fontsize=fs)
    axs_evs[5].set_xlabel('Imagine', fontsize=fs)


    fig_evs.savefig(f'./figures/figure_5_explained_variance.png')
    fig_evs.savefig(f'./figures/figure_5_explained_variance.svg')

    fig_pcs.tight_layout()
    fig_pcs.savefig(f'./figures/figure_5_first_component.png')
    print()

