from itertools import product

import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import colormaps
import matplotlib.cm as cm

from scipy import stats

from libs.mappings import color_map

PPTS = ['kh9', 'kh10', 'kh11', 'kh12', 'kh13', 'kh15', 'kh18', 'kh30']
NPCS, PPTS, FOLDS, DROPOUTS, REPEATS, METRICS = np.arange(6)
AUC = 0

# CMAP = 'plasma'

def get_data(path, type_):
    # data = [pcs, ppts, folds, metrics, train/test]    
    dirs = [d for d in path.iterdir() if d.is_dir() and 'pc' in d.stem]
    dirs = sorted(dirs, key=lambda x:int(x.stem[2:]))

    data = []
    for d in dirs:
        data += [np.stack([np.load(ppt/f'{type_}.npy') for ppt in d.iterdir()])]
    
    return np.stack(data)

def plot_panel_dropoff(ax, data):

    # cmap = colormaps['gnuplot']
    # cmap = colormaps[CMAP]
    cmap = cm.plasma

    dropout = data[:, :, :, :, :, AUC].transpose(NPCS, DROPOUTS, PPTS, FOLDS, REPEATS)
    dropout = dropout.reshape(dropout.shape[0], dropout.shape[1], -1)

    dropouts = np.arange(50, 101, 10)[::-1]
    pcs = np.concatenate([[3], np.arange(5, 51, 5)])

    colors = [cmap(int((pc/50)*200)) for pc in pcs]

    mean, std = dropout.mean(axis=-1), dropout.std(axis=-1)
    
    for i, (m, s) in enumerate(zip(mean, std)):
        ax.plot(dropouts, m, c=colors[i], label=pcs[i])
        ax.scatter(dropouts, m, s=25, zorder=2, facecolors=colors[i], edgecolors=colors[i])
        # ax.fill_between(dropouts, m-s, m+s, alpha=0.15, color='k', label='Standard deviation')
    
    ax.set_ylim(0.4, 1)
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.7)
    ax.invert_xaxis()

    ax.set_title('Decoding performance', fontsize='xx-large')
    ax.set_ylabel('Area under the curve', fontsize='x-large')
    ax.set_xlabel(f'% available channels', fontsize='x-large')

    ax.legend(frameon=False, fontsize=6)
    return ax

def plot_panel_distribution(ax, data):
    # adjust_to_lim = lambda x, lim: (x - (lim[1]-lim[0])) / (lim[1]-lim[0])

    # cmap = colormaps[CMAP]
    cmap = cm.plasma

    dropout = data[:, :, :, :, :, AUC].transpose(NPCS, DROPOUTS, PPTS, FOLDS, REPEATS)
    dropout = dropout.reshape(dropout.shape[0], dropout.shape[1], -1)

    pcs = np.concatenate([[3], np.arange(5, 51, 5)])

    colors = [cmap(int((pc/50)*200)) for pc in pcs]

    mean = dropout.mean(axis=-1)
    auc = mean.mean(axis=1)

    opt = np.argmax(auc)

    ax.scatter(pcs, auc, color=colors)  # TODO: color by colors of first panel
    ax.set_xticks(pcs)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel('N principal components', fontsize='x-large')
    ax.set_ylabel('Mean performance', fontsize='x-large')
    ax.set_title('Manifold stability', fontsize='xx-large')

    # ax.axvline(pcs[opt], color='grey', linestyle='--', label='Optimal performance')

    ax.annotate('Optimum', xy=(pcs[opt], auc[opt]+0.02), xytext=(pcs[opt], auc[opt]+.1),
                ha='center', fontsize='x-large',
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

    ylim = (.5, .9)
    ax.set_ylim(ylim)

    # ax.axvline(pcs[opt], 0, adjust_to_lim(auc[opt], ylim), color='black', linestyle='--', alpha=0.5, linewidth=1)
    return ax, opt


def plot_panel_mean_std(ax, data, opt):
    # cmap = colormaps[CMAP]
    cmap = cm.plasma

    pcs = np.concatenate([[3], np.arange(5, 51, 5)])
    colors = [cmap(int((pc/50)*200)) for pc in pcs]
    color = colors[opt]
    dropouts = np.arange(50, 101, 10)[::-1]

    dropout = data[:, :, :, :, :, AUC].transpose(NPCS, DROPOUTS, PPTS, FOLDS, REPEATS)

    dropout_opt = dropout[opt, :, :, :, :]
    
    mean = dropout_opt.mean(axis=FOLDS).mean(axis=-1)
    std =  dropout_opt.mean(axis=FOLDS).std(axis=-1)

    # ax.plot(dropouts, mean.mean(axis=-1), color=color)
    # ax.scatter(dropouts, mean.mean(axis=-1), color=color)

    # m, s = mean.mean(axis=-1), std.mean(axis=-1)
    # ax.fill_between(dropouts, m-s, m+s, alpha=0.15, color=color)

    ppt_colors = list(color_map().values())

    for i, (m, s) in enumerate(zip(mean.T, std.T)):
        ax.plot(dropouts, m, color=ppt_colors[i], alpha=1, linewidth=3)
        ax.fill_between(dropouts, m-s, m+s, color=ppt_colors[i], alpha=0.2)

    ax.invert_xaxis()
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.7)

    ax.set_title('Optimal components', fontsize='xx-large')
    ax.set_xlabel(f'% available channels', fontsize='x-large')
    ax.set_ylabel('Area under the curve', fontsize='x-large')
    ax.set_ylim(0.4, 1.0)

    return ax

def make(path):

    task = ('grasp') #, 'imagine')
    filter_ = ('beta') #, 'hg', 'betahg')

    data = get_data(path/task/filter_, 'dropout')

    N_COLS = 3
    N_ROWS = 1

    fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(15, 5))
    fig.suptitle('')

    fig.subplots_adjust(top=0.8)

    ax1 =      plot_panel_dropoff(axs[0], data)
    ax2, opt = plot_panel_distribution(axs[1], data)
    ax3 =      plot_panel_mean_std(axs[2], data, opt)
    # ax3 = plot_panel_std_per_ppt(axs[2], data, opt)

    for ax in [ax1, ax2, ax3]:
        # ax.set_aspect('square')
        ax.tick_params(axis='both', which='major', labelsize='large')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # fig.suptitle('Manifold stability by disabling channels', fontsize='xx-large')#, y=1.01)
    fig.savefig(r'./figures/figure_2_dropout.png')
    fig.savefig(r'./figures/figure_2_dropout.svg')


    return fig, axs