from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from libs import mappings

AUC = 0
I_NPCS, I_PPTS, I_FOLDS, I_DROPOUTS, I_REPEATS, I_METRICS = np.arange(6)

PCS = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
DROPOUTS = [100, 90, 80, 70 ,60, 50]
N_PPTS = len(mappings.PPTS)
COLORS_PC = [cm.plasma(int((pc/50)*200)) for pc in PCS]

def sort_by_key_and_participants(path):
    PC = -3
    PPT = -2

    pc_keys = [f'pc{pc}' for pc in PCS]
    ppt_keys = mappings.PPTS
    indices = list(product(pc_keys, ppt_keys))

    parts = path.parts
    return indices.index((parts[PC], parts[PPT]))


def get_data(main_path, type_):
    '''
    Returns AUCS of TEST: shape = [n_pcs, n_ppts, n_folds, n_repeats]
    '''
    paths = main_path.rglob(f'{type_}.npy')
    paths = sorted(paths, key=sort_by_key_and_participants)
    
    data = np.array([np.load(path)[:, :, :, AUC] for path in paths])  # shape = [10 x 6 x 10]
    data = data.reshape(-1, 8, *data.shape[1:])

    return data

def plot_panel_dropoff(ax, data):

    dropout = data.transpose(I_NPCS, I_DROPOUTS, I_PPTS, I_FOLDS, I_REPEATS)
    dropout = dropout.reshape(dropout.shape[0], dropout.shape[1], -1)

    mean = dropout.mean(axis=-1)
    
    for i, m in enumerate(mean):
        ax.plot(DROPOUTS, m, c=COLORS_PC[i], label=PCS[i])
        ax.scatter(DROPOUTS, m, s=25, zorder=2, facecolors=COLORS_PC[i], edgecolors=COLORS_PC[i])
    
    ax.set_ylim(0.4, 1)
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.7)
    ax.invert_xaxis()

    ax.set_title('Decoding performance', fontsize='xx-large')
    ax.set_ylabel('Area under the curve', fontsize='x-large')
    ax.set_xlabel(f'% available channels', fontsize='x-large')

    ax.legend(frameon=False, fontsize=6)
    return ax

def plot_panel_distribution(ax, data):

    dropout = data.transpose(I_NPCS, I_DROPOUTS, I_PPTS, I_FOLDS, I_REPEATS)
    dropout = dropout.reshape(dropout.shape[0], dropout.shape[1], -1)

    mean = dropout.mean(axis=-1)
    auc = mean.mean(axis=1)
    opt = np.argmax(auc)

    ax.scatter(PCS, auc, color=COLORS_PC)
    ax.set_xticks(PCS)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel('N principal components', fontsize='x-large')
    ax.set_ylabel('Mean performance', fontsize='x-large')
    ax.set_title('Manifold stability', fontsize='xx-large')

    ax.annotate('Optimum', xy=(PCS[opt], auc[opt]+0.02), xytext=(PCS[opt], auc[opt]+.1),
                ha='center', fontsize='x-large',
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

    ax.set_ylim(.5, .9)

    return ax, opt


def plot_panel_mean_std(ax, data, opt):

    dropout = data.transpose(I_NPCS, I_DROPOUTS, I_PPTS, I_FOLDS, I_REPEATS)
    dropout_opt = dropout[opt, :, :, :, :]
    
    mean = dropout_opt.mean(axis=I_FOLDS).mean(axis=-1)
    std =  dropout_opt.mean(axis=I_FOLDS).std(axis=-1)

    ppt_colors = list(mappings.color_map().values())

    for i, (m, s) in enumerate(zip(mean.T, std.T)):
        ax.plot(DROPOUTS, m, color=ppt_colors[i], alpha=1, linewidth=3)
        ax.fill_between(DROPOUTS, m-s, m+s, color=ppt_colors[i], alpha=0.2)

    ax.invert_xaxis()
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.7)

    ax.set_title('Optimal components', fontsize='xx-large')
    ax.set_xlabel(f'% available channels', fontsize='x-large')
    ax.set_ylabel('Area under the curve', fontsize='x-large')
    ax.set_ylim(0.4, 1.0)

    return ax

def make(path):

    task = ('grasp', 'imagine')
    filter_ = ('beta', 'hg', 'betahg')

    data = get_data(path/task/filter_, 'dropout')

    N_COLS = 3
    N_ROWS = 1

    fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(15, 5))
    fig.suptitle('')

    fig.subplots_adjust(top=0.8)

    ax1 =      plot_panel_dropoff(axs[0], data)
    ax2, opt = plot_panel_distribution(axs[1], data)
    ax3 =      plot_panel_mean_std(axs[2], data, opt)

    for ax in [ax1, ax2, ax3]:

        ax.tick_params(axis='both', which='major', labelsize='large')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    fig.savefig(r'./figures/figure_2_dropout.png')
    fig.savefig(r'./figures/figure_2_dropout.svg')

    return fig, axs