import numpy as np
import matplotlib.cm as cm

from matplotlib.patches import Rectangle
from pyriemann.estimation import Covariances
from scipy.stats import mannwhitneyu

MAX_PCS = 50
COVMATS, LABELS = 0, 1

def get_sig_level(p, n_tests=1):

    p *= n_tests # Bonferonni

    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'

    return 'n.s.'

def estimate_covariance_matrices(x):
    x = x.transpose(0, 2, 1)
    return Covariances(estimator='lwf').transform(x)

def load_covariance_matrices(path):
    sort_by_parent_dir = lambda p: p.parent.name

    data =   [np.load(file) for file in sorted(path.rglob('processed_data.npy'), key=sort_by_parent_dir)]
    labels = [np.load(file) for file in sorted(path.rglob('labels.npy'),         key=sort_by_parent_dir)]

    move, rest = [], []
    for eeg, class_labels in zip(data, labels):
        
        covs = estimate_covariance_matrices(eeg)
        is_move = np.where(class_labels!='rest', True, False)
        
        move.append(covs[is_move])
        rest.append(covs[~is_move])

    return np.vstack(move), np.vstack(rest)

def plot_panel(ax, move, rest):

    color = cm.plasma(0)

    move, rest = move[:, 0, 0], rest[:, 0, 0]

    _, p = mannwhitneyu(rest, move, alternative='two-sided')

    # Plot
    vp_rest = ax.violinplot(rest, positions=[0], showmeans=False, showmedians=True, showextrema=False)
    vp_move = ax.violinplot(move, positions=[1], showmeans=False, showmedians=True, showextrema=False)

    for vp in [vp_rest, vp_move]:
        vp['cmedians'].set_color('black')
        vp['bodies'][0].set_facecolor(color)
        vp['bodies'][0].set_edgecolor(color)
        vp['bodies'][0].set_alpha(.5)

    xlim = ax.get_xlim()
    ax.add_patch(Rectangle((xlim[0], 0), np.abs(xlim[0])-0.01, max(rest), color='white'))
    ax.add_patch(Rectangle((.5, 0), 0.49, max(move), color='white'))

    ax.set_yscale('log')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Rest', 'Move'], fontsize='x-large')
    ax.set_xlim(-0.325, 1.325)

    # Significance line
    height = np.max([rest.max(), move.max()]) * 1.1
    offset = height*0.1
    xmin, xmax = 0, 1

    is_sig = get_sig_level(p, n_tests=1)

    xlim = ax.get_xlim()
    l = np.abs(xlim).sum()
    xmin_frac = (xmin + np.abs(xlim[0]))/l
    xmax_frac = (xmax + np.abs(xlim[0]))/l

    ax.axhline(height, xmin_frac, xmax_frac, color='k', linewidth=2)
    ax.annotate(f'{is_sig} (p={p:.3f})', 
                xy=((xmax+xmin)/2, height), 
                xytext=((xmax+xmin)/2, height+offset),
                ha='center',
                va='bottom',
                fontsize='xx-large')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_ylabel('Variance in PC 1', fontsize='xx-large')

    return ax