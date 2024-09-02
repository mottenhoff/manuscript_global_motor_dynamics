import numpy as np
import matplotlib.pyplot as plt

from libs.mappings import color_map, PPTS

cmap = color_map()
AUC, BAC = 0, 1
TRAIN, TEST = 0, 1
PCS = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
N_FOLDS = 10

def load_cross(path):

    gti = [np.load(file/'results_test.npy') for file in sorted(path.rglob('grasp-to-imagine'))]
    itg = [np.load(file/'results_test.npy') for file in sorted(path.rglob('imagine-to-grasp'))]

    return np.stack(gti), np.stack(itg)

def load_within(path, task, filter_):
    
    sort_parent_by_numbers_in_string = lambda x: int(x.parent.name \
                                                      .lower() \
                                                      .strip('abcdefghijklmnopqrstuvwxyz'))

    files = list((path/task/filter_).rglob('full.npy'))

    within = np.empty((N_FOLDS, len(PPTS), len(PCS)))
    for i_pcs, pc in enumerate(PCS):
        pc_files = [file for file in files if f'pc{pc}' == file.parts[-3]]
        pc_files = sorted(pc_files, key=sort_parent_by_numbers_in_string)
        
        for i_ppts, file in enumerate(pc_files):
            within[:, i_ppts, i_pcs] = np.load(file)[:, AUC, TEST]

    return within

def plot_panel_individual(ax, gti, itg):
    
    for ppt in range(gti.shape[0]):

        ax.plot(PCS, gti[ppt, :, AUC], linestyle='solid',  color=cmap[PPTS[ppt]])
        ax.plot(PCS, itg[ppt, :, AUC], linestyle='dashed', color=cmap[PPTS[ppt]])

    return ax

def plot_panel_average(ax, gw, iw, gti, itg):

    gti_mean, gti_std = gti[:, :, AUC].mean(axis=0), gti[:, :, AUC].std(axis=0)
    itg_mean, itg_std = itg[:, :, AUC].mean(axis=0), itg[:, :, AUC].std(axis=0)
    
    ax.plot(PCS, gti_mean, linestyle='solid')
    ax.fill_between(PCS, gti_mean-gti_std, gti_mean+gti_std, alpha=0.3)
    
    ax.plot(PCS, itg_mean, linestyle='dashed')
    ax.fill_between(PCS, itg_mean-itg_std, itg_mean+itg_std, alpha=0.3)


    return ax

def compare_runs(now, prev):
    '''
    There is some numerical instabillity or randomness within the covariance estimation.
    Even when a seed is set, the values slightly vary. 
    
    '''
    now1 = now[0, :, AUC]
    prev1 = prev[:, 0]
    now2 = np.load(r'results\20240823_1007\beta\p1\grasp-to-imagine\results_test.npy')[:, AUC]

    now3seed1 = np.load(r'results\20240823_1024\beta\p1\grasp-to-imagine\results_test.npy')[:, AUC]
    now3seed2 = np.load(r'results\20240823_1031\beta\p1\grasp-to-imagine\results_test.npy')[:, AUC]
    return

def make(path):
    
    filters = ('beta', 'hg', 'betahg')
    tasks = ('grasp-to-imagine', 'imagine-to-grasp')

    n_rows = 2
    n_cols = 3

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    
    for col in range(n_cols):

        gti, itg = load_cross(path/filters[col])
        gw = load_within(path.parent/'full_run', 'grasp', filters[col])
        iw = load_within(path.parent/'full_run', 'imagine', filters[col])
        # compare_runs(gti, np.load('cross_prev.npy'))  # Debug

        axs[0, col] = plot_panel_individual(axs[0, col], gti, itg)
        axs[0, col].set_title(filters[col])

        axs[1, col] = plot_panel_average(axs[1, col], gti, itg)
        axs[1, col].set_title('mean+std')

    for ax in axs.ravel():
        ax.set_xticks(PCS)
        ax.set_xlim(0, 50)
        ax.set_ylim(0.4, 1)
        ax.axhline(0.5, linestyle='--', color='black', alpha=0.7)

    plt.show(block=True)

    return
