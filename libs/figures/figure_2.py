from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from libs import mappings

FONTSIZE = 25

PCS = 0
PPTS = 1
FOLDS = 2

AUC = 0
BAC = 1
TRAIN = 0
TEST = 1

def fdrcorrection(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def annotate_min_max(ax):

    line = ax.lines[0]
    x_values, y_values = line.get_xdata(), line.get_ydata()
    x, y = x_values[np.argmax(y_values)], np.max(y_values)
    # max_y, idx_max_y = np.max(y_values), np.argmax(y_values)

    ax.annotate(np.round(y, 2), (x, y), xytext=(x-1, y+0.05), ha='center', fontsize=FONTSIZE/2)

    return ax

def get_data(path, type_):
    # data = [pcs, ppts, folds, metrics, train/test]    
    dirs = [d for d in path.iterdir() if d.is_dir() and 'pc' in d.stem]
    dirs = sorted(dirs, key=lambda x:int(x.stem[2:]))

    data = []
    for d in dirs:
        data += [np.stack([np.load(ppt/f'{type_}.npy') for ppt in d.iterdir()])]
    
    return np.stack(data)

def plot_panel(ax, data):

    pcs = np.concatenate([[3], np.arange(5, 51, 5)])

    aucs = data[:, :, :, AUC, TEST].reshape(data.shape[0], np.multiply(*data.shape[1:3]))

    mean, std = aucs.mean(axis=1), aucs.std(axis=1)

    t, p = stats.ttest_1samp(aucs, .5, axis=1)
    p = fdrcorrection(p)

    facecolors = np.where(p<0.05, 'black', 'lightgrey')
    edgecolors = ['black'] * facecolors.size

    ax.plot(pcs, mean, color='k', zorder=1)
    ax.scatter(pcs, mean, facecolors=facecolors, edgecolors=edgecolors, s=25, zorder=2)
    ax.fill_between(pcs, mean-std, mean+std, alpha=0.15, color='k')

    ax = annotate_min_max(ax)

    colors = [mappings.color_map()[f'p{i}'] for i in np.arange(data.shape[PPTS]) + 1]

    for ppt in np.arange(data.shape[PPTS]):
        ax.plot(pcs, data[:, ppt, :, AUC, TEST].mean(axis=1), color=colors[ppt], alpha=0.3)

    ax.axhline(0.5, alpha=0.5, linestyle='dashed', color='k')
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 1)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(pcs)
    return ax

def plot_specific_performance(ax, data, selection='max'):

    n_ppts = np.arange(data.shape[1])
    pcs = np.concatenate([[3], np.arange(5, 51, 5)])

    aucs = data[:, :, :, AUC, TEST].reshape(data.shape[0], data.shape[1], -1)

    if selection == 'max':
        npcs_idx = np.argmax(aucs.reshape(aucs.shape[0], -1).mean(axis=1))
    else:
        npcs_idx = np.where(pcs==selection)[0].squeeze()

    mean, std = aucs[npcs_idx, :, :].mean(axis=-1), aucs[npcs_idx, :, :].std(axis=-1)

    colors = [mappings.color_map()[f'p{i}'] for i in n_ppts+1]
    ax.bar(n_ppts, mean, zorder=2, color=colors)
    ax.errorbar(n_ppts, mean, yerr=std, fmt='o', capsize=5, color='black', zorder=2)

    ax.axhline(0.5, color='black', linestyle='dashed', alpha=0.5) #, linewidth=1, zorder=2)

    ax.set_xticks(n_ppts)
    ax.set_xticklabels(n_ppts+1)

    # Stylize
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.grid(visible=True, axis='y', linewidth=1, linestyle='dotted', color='grey', zorder=1)

    # if selection == 'max':
    #     ax.text(0.11, 1, f'N PCs={pcs[npcs_idx]}', 
    #             ha='center', va='center', transform=ax.transAxes)
        # ax.set_title('Individual scores at\nmax performance', fontsize='xx-large')
    # else:
        # ax.set_title(f'Individual scores at {pcs[npcs_idx]} components', fontsize='xx-large')
    # ax.set_title(f'Scores at max performance (={pcs[npcs_idx]}')


    return ax

def make(path):

    tasks = ('grasp', 'imagine')
    filters = ('beta', 'hg', 'betahg')

    selection = 10 #'max'

    data = [get_data(path/task/filters, 'full') \
            for task, filters in product(tasks, filters)]

    N_COLS = 3
    N_ROWS = 3

    ax_idc = list(product(np.arange(N_ROWS), np.arange(N_COLS)))

    fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(16, 9))
    fig.suptitle('')

    ax_eb =   plot_panel(axs[0, 0], data[0])
    ax_ehg =  plot_panel(axs[1, 0], data[1])
    ax_ebhg = plot_panel(axs[2, 0], data[2])
    ax_ib =   plot_panel(axs[0, 1], data[3])
    ax_ihg =  plot_panel(axs[1, 1], data[4])
    ax_ibhg = plot_panel(axs[2, 1], data[5])

    ax_bar_b =   plot_specific_performance(axs[0, 2], data[0], selection=selection)
    ax_bar_hg =  plot_specific_performance(axs[1, 2], data[1], selection=selection)
    ax_bar_bhg = plot_specific_performance(axs[2, 2], data[2], selection=selection)

    if selection == 'max':
        ax_bar_b.set_title('Individual scores at\nmax performance', fontsize='xx-large')
    else:
        ax_bar_b.set_title(f'Individual scores at\n{selection} components', fontsize='xx-large')

    ax_eb.set_title('Execute', fontsize=20)
    ax_ib.set_title('Imagine', fontsize=20)

    ## xx-large = 17.28
    ax_eb.set_ylabel('Beta', fontsize=20)
    ax_ehg.set_ylabel('High-gamma', fontsize=20)
    ax_ebhg.set_ylabel('Both', fontsize=20)

    ax_bar_bhg.set_xlabel('Participant', fontsize='x-large')
    ax_ebhg.set_xlabel('Principal components', fontsize='x-large')
    ax_ibhg.set_xlabel('Principal components', fontsize='x-large')

    ax_bar_b.yaxis.set_label_coords(-.2, 0.5)
    ax_bar_b.text(-.12, .5, '', transform=ax_bar_b.transAxes, 
                rotation='vertical', va='center', ha='center', fontsize='large')

    ax_bar_hg.yaxis.set_label_coords(-.2, 0.5)
    ax_bar_hg.text(-.12, .5, '', transform=ax_bar_hg.transAxes, 
                rotation='vertical', va='center', ha='center', fontsize='large')

    ax_bar_bhg.yaxis.set_label_coords(-.2, 0.5)
    ax_bar_bhg.text(-.12, .5, 'Area under the curve', transform=ax_bar_bhg.transAxes, 
                rotation='vertical', va='center', ha='center', fontsize='x-large')

    fig.subplots_adjust(hspace=0.3)
    fig.savefig(r'./figures/figure_1_general_overview.png')
    fig.savefig(r'./figures/figure_1_general_overview.svg')

    return fig, axs

