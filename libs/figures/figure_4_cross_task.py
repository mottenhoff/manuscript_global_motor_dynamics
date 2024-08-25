import numpy as np
import matplotlib.cm as cm

from libs import mappings

FONTSIZE = 25

PCS = 0
PPTS = 1
FOLDS = 2

AUC = 0
BAC = 1
TRAIN = 0
TEST = 1

PCS = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
N_PCS = len(PCS)
N_PPTS = len(mappings.PPTS)
COLORS = {'exec':  cm.plasma(0),    # Rest in fig 1
          'imag':  cm.plasma(127)}  # Move in fig 1


def fdrcorrection(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    # p = np.asfarray(p)
    p = np.array(p, dtype=np.float64)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def annotate_min_max(ax):

    line = ax.lines[0]
    x_values, y_values = line.get_xdata(), line.get_ydata()
    x, y = x_values[np.argmax(y_values)], np.max(y_values)

    ax.annotate(np.round(y, 2), (x, y), xytext=(x-1, y+0.05), fontsize=FONTSIZE/2)

    return ax

def plot_panel(ax, data_cross, data_imag, data_exec):

    shape_exec = data_exec.shape
    shape_imag = data_imag.shape

    # STD calculation here is different than the manuscript. Here is it the std over all folds,
    #     in the paper its the std over the means, resulting in a smaller std in the paper
    # See this line: np.std([np.mean(aucs[:, s:e]) for s, e in zip(np.arange(0, 72, 8), np.arange(8, 80, 8))])
    # mean, std = aucs.mean(axis=1), aucs.std(axis=1)

    # t, p = stats.ttest_1samp(aucs, .5, axis=1)
    # p = fdrcorrection(p)

    # ax = annotate_min_max(ax)
    # facecolors = np.where(p<0.05, 'black', 'lightgrey')
    # edgecolors = ['black'] * facecolors.size

    gti_mean = data_cross['grasp-to-imagine'].mean(axis=0)[:, AUC]
    itg_mean = data_cross['imagine-to-grasp'].mean(axis=0)[:, AUC]
    exec_mean = data_exec.reshape(shape_exec[0], -1).mean(axis=1)
    imag_mean = data_imag.reshape(shape_imag[0], -1).mean(axis=1)


    ax.plot(PCS, gti_mean, linestyle='solid', color=COLORS['imag'], label='Execute to Imagine')
    ax.scatter(PCS, gti_mean, facecolors=COLORS['imag'], edgecolors=COLORS['imag'], s=25, zorder=2)

    ax.plot(PCS, itg_mean, linestyle='solid', color=COLORS['exec'], label='Imagine to Execute')
    ax.scatter(PCS, itg_mean, facecolors=COLORS['exec'], edgecolors=COLORS['exec'], s=25, zorder=2)

    ax.plot(PCS, exec_mean, linestyle='dashed', color=COLORS['exec'],label='Execute to Execute')
    ax.scatter(PCS, exec_mean, facecolors=COLORS['exec'], edgecolors=COLORS['exec'], s=25, zorder=2)

    ax.plot(PCS, imag_mean, linestyle='dashed',color=COLORS['imag'], label='Imagine to Imagine')
    ax.scatter(PCS, imag_mean, facecolors=COLORS['imag'], edgecolors=COLORS['imag'], s=25, zorder=2)

    ax.axhline(0.5, linestyle='--', color='black', alpha=0.7)
    
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlabel('Principal components', fontsize='xx-large')
    ax.set_ylabel('Area under the curve', fontsize='xx-large')
    ax.set_xticks(PCS)
    ax.set_xlim(0, 50)
    ax.set_ylim(0.4, 1)
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax