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

EXEC_BETA, EXEC_HG, EXEC_BHG = 0, 1, 2
IMAG_BETA, IMAG_HG, IMAG_BGH = 3, 4, 5

PCS = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
N_PCS =  len(PCS)
N_PPTS = len(mappings.PPTS)
N_FOLDS = 10


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

    ax.annotate(np.round(y, 2), (x, y), xytext=(x-1, y+0.05), ha='center', fontsize=FONTSIZE/2)

    return ax

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
    Returns AUCS of TEST: shape = [n_pcs, n_ppts, n_folds]
    '''
    paths = main_path.rglob(f'{type_}.npy')
    paths = sorted(paths, key=sort_by_key_and_participants)
    
    data = np.array([np.load(path)[:, AUC, TEST] for path in paths])
    data = data.reshape(-1, N_PPTS, N_FOLDS)

    return data

def plot_panel(ax, data):

    aucs = data.reshape(data.shape[0], -1)
    mean, std = aucs.mean(axis=1), aucs.std(axis=1)

    _, p = stats.ttest_1samp(aucs, .5, axis=1)
    p = fdrcorrection(p)

    facecolors = np.where(p<0.05, 'black', 'lightgrey')
    edgecolors = ['black'] * facecolors.size

    ax.plot(PCS, mean, color='k', zorder=1)
    ax.scatter(PCS, mean, facecolors=facecolors, edgecolors=edgecolors, s=25, zorder=2)
    ax.fill_between(PCS, mean-std, mean+std, alpha=0.15, color='k')

    ax = annotate_min_max(ax)

    colors = [mappings.color_map()[f'p{i}'] for i in np.arange(data.shape[PPTS]) + 1]

    for ppt in np.arange(data.shape[PPTS]):
        ax.plot(PCS, data[:, ppt, :].mean(axis=1), color=colors[ppt], alpha=0.3)

    ax.axhline(0.5, alpha=0.5, linestyle='dashed', color='k')
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 1)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(PCS)
    return ax

def plot_specific_performance(ax, data, selection='max'):

    ppts = np.arange(N_PPTS)
    aucs = data.reshape(data.shape[0], data.shape[1], -1)
    colors = [mappings.color_map()[ppt] for ppt in mappings.PPTS]

    if selection == 'max':
        npcs_idx = np.argmax(aucs.reshape(aucs.shape[0], -1).mean(axis=1))
    else:
        npcs_idx = PCS.index(selection)

    mean, std = aucs[npcs_idx, :, :].mean(axis=-1), aucs[npcs_idx, :, :].std(axis=-1)

    ax.bar(ppts, mean, zorder=2, color=colors)
    ax.errorbar(ppts, mean, yerr=std, fmt='o', capsize=5, color='black', zorder=2)
    ax.axhline(0.5, color='black', linestyle='dashed', alpha=0.5) #, linewidth=1, zorder=2)

    ax.set_xticks(np.arange(N_PPTS))
    ax.set_xticklabels(mappings.PPTS)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.grid(visible=True, axis='y', linewidth=1, linestyle='dotted', color='grey', zorder=1)

    return ax

def check_stat_diff_eb_max_ehg_max(eb, ehg):
    IDX_OF_MAX_EXEC_BETA = -3
    IDX_OF_MAX_EXEC_HG = -1

    eb_max = eb[IDX_OF_MAX_EXEC_BETA, :, :]
    ehg_max = ehg[IDX_OF_MAX_EXEC_HG, :, :]
    
    print(stats.ttest_ind(eb_max.flatten(), ehg_max.flatten()))

def check_stat_diff_e_i(data):

    groups = ['beta', 'high_gamma', 'betahighgamma']
    groups_to_compare = [[EXEC_BETA, IMAG_BETA],
                         [EXEC_HG,   IMAG_HG],
                         [EXEC_BHG,  IMAG_BGH]]

    for i, (exec_idx, imag_idx) in enumerate(groups_to_compare):

        eb = data[exec_idx]
        ib = data[imag_idx]

        # Select data and make it shape ppts x pcs
        eb = eb.mean(axis=-1).T
        ib = ib.mean(axis=-1).T

        with open(f'/figures/exec_vs_imag_{groups[i]}_stats.txt', 'w') as f:
                
            # Step 1: Perform one-way ANOVA
            F, p = stats.f_oneway(eb.flatten(), ib.flatten())
            print("Overall ANOVA:", file=f)
            print("F-statistic:", F, file=f)
            print("p-value:", p, file=f)

            # Step 2: If significant overall difference, conduct pairwise t-tests for each principal component
            if p < 0.05:
                print("\nPairwise t-tests for each principal component:", file=f)
                for pc in range(eb.shape[1]):
                    t_statistic, p_value = stats.ttest_ind(eb[:, pc], ib[:, pc])
                    p_value *= eb.shape[1]  # Bonferroni correction
                    print(f"Principal Component {pc+1}: t-statistic = {t_statistic}, p-value = {p_value}, <0.05={True if p_value < 0.05 else False}", file=f)


def make(path):

    tasks = ('grasp', 'imagine')
    filters = ('beta', 'hg', 'betahg')

    selection = 10

    data = [get_data(path/task/filters, 'full') \
            for task, filters in product(tasks, filters)]

    check_stat_diff_eb_max_ehg_max(data[EXEC_BETA], data[EXEC_HG])
    check_stat_diff_e_i(data)

    N_COLS = 4
    N_ROWS = 3

    fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(20, 9))
    fig.suptitle('')

    ax_eb =   plot_panel(axs[0, 0], data[EXEC_BETA])
    ax_ehg =  plot_panel(axs[1, 0], data[EXEC_HG])
    ax_ebhg = plot_panel(axs[2, 0], data[EXEC_BHG])
    ax_ib =   plot_panel(axs[0, 2], data[IMAG_BETA])
    ax_ihg =  plot_panel(axs[1, 2], data[IMAG_HG])
    ax_ibhg = plot_panel(axs[2, 2], data[IMAG_BGH])

    ax_bar_b =   plot_specific_performance(axs[0, 1], data[EXEC_BETA], selection=selection)
    ax_bar_hg =  plot_specific_performance(axs[1, 1], data[EXEC_HG], selection=selection)
    ax_bar_bhg = plot_specific_performance(axs[2, 1], data[EXEC_BHG], selection=selection)

    ax_bar_ib =   plot_specific_performance(axs[0, 3], data[IMAG_BETA], selection=selection)
    ax_bar_ihg =  plot_specific_performance(axs[1, 3], data[IMAG_HG], selection=selection)
    ax_bar_ibhg = plot_specific_performance(axs[2, 3], data[IMAG_BGH], selection=selection)

    if selection == 'max':
        ax_bar_b.set_title('Individual scores at\nmax performance', fontsize='xx-large')
    else:
        ax_bar_b.set_title( f'Execute\nIndividual scores at\n{selection} components ', fontsize='xx-large')
        ax_bar_ib.set_title(f'Imagine\nIndividual scores at\n{selection} components ', fontsize='xx-large')

    ax_eb.set_title('Execute', fontsize=20)
    ax_ib.set_title('Imagine', fontsize=20)

    ## xx-large = 17.28
    ax_eb.set_ylabel('Beta', fontsize=20)
    ax_ehg.set_ylabel('High-gamma', fontsize=20)
    ax_ebhg.set_ylabel('Both', fontsize=20)

    ax_bar_bhg.set_xlabel('Participant', fontsize='x-large')
    ax_bar_ibhg.set_xlabel('Participant', fontsize='x-large')
    ax_ebhg.set_xlabel('Principal components', fontsize='x-large')
    ax_ibhg.set_xlabel('Principal components', fontsize='x-large')

    ax_bar_b.yaxis.set_label_coords(-.2, 0.5)
    ax_bar_b.text(-.12, .5, '', transform=ax_bar_b.transAxes, 
                rotation='vertical', va='center', ha='center', fontsize='large')

    ax_bar_hg.yaxis.set_label_coords(-.2, 0.5)
    ax_bar_hg.text(-.12, .5, '', transform=ax_bar_hg.transAxes, 
                rotation='vertical', va='center', ha='center', fontsize='large')

    ax_bar_bhg.yaxis.set_label_coords(-.2, 0.5)
    ax_bar_bhg.text(-.13, .5, 'Area under the curve', transform=ax_bar_bhg.transAxes, 
                rotation='vertical', va='center', ha='center') # , fontsize='x-large'

    fig.subplots_adjust(hspace=0.3)
    fig.savefig(r'./figures/figure_1_general_overview.png')
    fig.savefig(r'./figures/figure_1_general_overview.svg')

    return fig, axs