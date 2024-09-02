import numpy as np
import matplotlib.cm as cm
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

PCS = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
N_PCS = len(PCS)
N_PPTS = len(mappings.PPTS)
COLORS = {'exec':  cm.plasma(0),
          'imag':  cm.plasma(127),
          'lightgrey': np.array((211/255, 211/255, 211/255, 1.0))} 



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

def get_scatter_colors(p_values, task):
    facecolors = np.where(p_values < .05)
    facecolors = np.array([COLORS[task] if p_value <0.05 else COLORS['lightgrey'] for p_value in p_values])
    edgecolors = np.tile(np.array(COLORS[task]), (facecolors.size, 1))
    return facecolors, edgecolors

def plot_panel(ax, data_cross, data_imag, data_exec):

    shape_exec = data_exec.shape
    shape_imag = data_imag.shape

    # Check if above chance
    # note that within task performance has 80 samples here,
    # while between task has only 8 samples. Because no cross
    # validation for cross task.
    p_values = []
    for data in [data_cross['grasp-to-imagine'][:, :, AUC].T,
                 data_cross['imagine-to-grasp'][:, :, AUC].T,
                 data_exec.reshape(shape_exec[0], -1),
                 data_imag.reshape(shape_imag[0], -1)]:
        t, p = stats.ttest_1samp(data, .5, axis=1)
        p = fdrcorrection(p)
        p_values.append(p)

    p_values = np.array(p_values)

    print('is exec-to-imag sig diff than imag-to-imag?')
    F, p = stats.f_oneway(data_imag.flatten(), 
                          data_cross['grasp-to-imagine'][:, :, AUC].flatten())
    print(f'one-way anova: p={p}')

    if p<0.05:
        for pc in range(data_imag.shape[0]):
            t, p = stats.ttest_ind(data_imag[pc, :, :].flatten(),
                                   data_cross['grasp-to-imagine'][:, pc, AUC])
            print(PCS[pc], p*N_PCS<0.05, p*N_PCS)

    print('is imag-to-exec sig diff than imag-to-imag?')
    F, p = stats.f_oneway(data_exec.flatten(), 
                          data_cross['imagine-to-grasp'][:, :, AUC].flatten())
    print(f'one-way anova: p={p}')

    if p<0.05:
        for pc in range(data_exec.shape[0]):
            t, p = stats.ttest_ind(data_exec[pc, :, :].flatten(),
                                   data_cross['imagine-to-grasp'][:, pc, AUC])
            print(PCS[pc], p*N_PCS<0.05, p*N_PCS)

    gti_mean = data_cross['grasp-to-imagine'].mean(axis=0)[:, AUC]
    itg_mean = data_cross['imagine-to-grasp'].mean(axis=0)[:, AUC]
    exec_mean = data_exec.reshape(shape_exec[0], -1).mean(axis=1)
    imag_mean = data_imag.reshape(shape_imag[0], -1).mean(axis=1)


    ax.plot(PCS, gti_mean, linestyle='solid', color=COLORS['imag'], label='Execute to Imagine')
    facecolors, edgecolors = get_scatter_colors(p_values[0, :], 'imag')
    ax.scatter(PCS, gti_mean, facecolors=facecolors, edgecolors=edgecolors, s=25, zorder=2)

    ax.plot(PCS, itg_mean, linestyle='solid', color=COLORS['exec'], label='Imagine to Execute')
    facecolors, edgecolors = get_scatter_colors(p_values[1, :], 'exec')
    ax.scatter(PCS, itg_mean, facecolors=facecolors, edgecolors=edgecolors, s=25, zorder=2)

    ax.plot(PCS, exec_mean, linestyle='dashed', color=COLORS['exec'],label='Execute to Execute')
    facecolors, edgecolors = get_scatter_colors(p_values[2, :], 'exec')
    ax.scatter(PCS, exec_mean, facecolors=facecolors, edgecolors=edgecolors, s=25, zorder=2)

    ax.plot(PCS, imag_mean, linestyle='dashed',color=COLORS['imag'], label='Imagine to Imagine')
    facecolors, edgecolors = get_scatter_colors(p_values[3, :], 'imag')
    ax.scatter(PCS, imag_mean, facecolors=facecolors, edgecolors=edgecolors, s=25, zorder=2)

    ax.axhline(0.5, linestyle='--', color='black', alpha=0.7)
    
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlabel('Principal components', fontsize='xx-large')
    ax.set_ylabel('Area under the curve', fontsize='xx-large')
    ax.set_title('Cross-task', fontsize='xx-large')
    ax.set_xticks(PCS)
    ax.set_xlim(0, 50)
    ax.set_ylim(0.4, 1)
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax